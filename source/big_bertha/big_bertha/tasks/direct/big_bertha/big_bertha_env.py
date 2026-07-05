# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# spdrbot3_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from .big_bertha_env_cfg import BigberthaEnvCfg


class BigberthaEnv(DirectRLEnv):
    cfg: BigberthaEnvCfg

    def __init__(self, cfg: BigberthaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity + yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Optional fixed-command override (shape (3,) [vx, vy, omega]). When set
        # (e.g. by play_fixed_vel.py), _reset_idx applies it INSTEAD of sampling
        # random commands, so explicit zero commands keep the robot stationary
        # instead of inheriting a random forward command on every reset (#40).
        self._command_override = None

        # Per-dim observation noise std (lazily built on device in _get_observations)
        self._obs_noise_std = None

        # Get specific body indices
        # preserve_order=True so the returned order matches this list exactly:
        # arm_c_1=FR(idx0), arm_c_2=FL(idx1), arm_c_3=RL(idx2), arm_c_4=RR(idx3),
        # verified against the foot positions in the base frame from the URDF.
        self._feet_ids, _ = self._contact_sensor.find_bodies(
            ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"], preserve_order=True
        )
        # Articulation body indices for the SAME feet (body ordering can differ
        # from the contact sensor), used to read world-frame foot height z for
        # the swing-foot clearance reward. Same ordered name list => aligned with
        # _feet_ids per-foot, which the clearance gate relies on.
        self._feet_body_ids, _ = self._robot.find_bodies(
            ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"], preserve_order=True
        )
        # Foot CONTACT-POINT offset in the arm_c link frame. The arm_c link origin
        # sits at the knee (~0.145 m above the contact), so its world velocity is
        # contaminated by body motion and is NOT a valid slip signal. This offset
        # (from the arm_c collision mesh origin in the URDF) maps the link pose to
        # the actual foot tip; verified in-sim (planted tip z = 0.000). Used by the
        # foot-tip slip penalty in _get_rewards.
        self._tip_offset = torch.tensor([0.01, -0.145, -0.053], device=self.device)
        # Body +x unit vector, for projecting foot motion onto the heading
        # direction in the forward-stride reward.
        self._x_unit = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        # Event-based stride state: the world tip position at each foot's last
        # LIFTOFF, and the previous-step contact mask. The stride reward pays a foot
        # only at TOUCHDOWN, for how far forward it moved since liftoff -- so it
        # cannot be earned by sweeping the foot forward while in contact.
        self._foot_liftoff_pos = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._prev_contact = torch.ones(self.num_envs, 4, dtype=torch.bool, device=self.device)
        # v0.9 gait clock: cycle phase in [0,1) advanced at cfg.gait_frequency, plus
        # fixed per-foot offsets defining a WAVE crawl. Feet order = _feet_ids order
        # [FR, FL, RL, RR]; offsets stagger the swing windows one-at-a-time
        # (sequence RR -> FL -> RL -> FR over a cycle) and the FR/FL + RL/RR pairs
        # differ by exactly 0.5, which keeps the mirror-symmetry augmentation valid
        # (mirroring = swapping clock dims within each pair, see symmetry.py).
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._gait_offsets = torch.tensor([0.0, 0.5, 0.25, 0.75], device=self.device)
        self._die_body_ids, _ = self._contact_sensor.find_bodies(["arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"])
        # Articulation base index + buffers for the SUSTAINED lateral-bias DR: a
        # constant body-frame sideways force + yaw torque held for the whole
        # episode (set_external_force_and_torque applies in the LOCAL link frame,
        # is_global=False). This is the systematic disturbance a pure velocity
        # penalty could not supply -- the policy must actively walk straight
        # against a DART-like crab/yaw push, building the lateral+yaw authority
        # that transfers. Magnitudes sampled per-episode in _reset_idx.
        self._robot_base_id, _ = self._robot.find_bodies("base_link")
        self._lat_bias = torch.zeros(self.num_envs, device=self.device)
        self._yaw_bias = torch.zeros(self.num_envs, device=self.device)
        self._ext_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._ext_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Diagonal foot pairs for the trot: {FR,RL}={0,2} swing together while
        # {FL,RR}={1,3} are in stance, then swap. (Was [[0,3],[1,2]] = same-side
        # legs, which let the policy satisfy the gait reward with a pronk.)
        self._foot_pairs = [[0, 2], [1, 3]]  # [FR+RL, FL+RR] diagonals

        # Identity negate — ImuCfg offset is identity, so sensor frame =
        # base_link and IMU data is already in base_link frame natively.
        self._imu_negate = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "joint_deviation",
                "base_height",
                "joint_activity",
                "feet_air_time",
                "crawl_gait",
                "gait_stance_still",
                "gait_swing_unload",
                "foot_clearance",
                "multi_swing_pen",
                "track_ang_vel_z_exp",
                "yaw_progress",
                "yaw_straight_pen",
                "lat_straight_pen",
                "stand_still_pen",
                "forward_progress",
                "foot_dragging",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Advance the gait clock once per policy step (50 Hz).
        self._gait_phase = (self._gait_phase + self.cfg.gait_frequency * self.step_dt) % 1.0
        self._actions = torch.clamp(actions.clone(), -1.0, 1.0)
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        # Action (joint-target) noise: the deployed gazebo joints track the
        # position targets through an explicit effort-PD that overshoots/lags
        # vs Isaac's implicit actuator, so the realised joint trajectory differs
        # from the commanded one. Perturbing the target here (unobserved) forces
        # the policy to learn a gait that is stable under imperfect tracking
        # instead of a fragile, Isaac-exact one that drifts/veers in gazebo.
        if self.cfg.action_noise_std > 0.0:
            self._processed_actions = self._processed_actions + (
                torch.randn_like(self._processed_actions) * self.cfg.action_noise_std
            )
        # Hold the per-episode sustained bias: body-y force (lateral crab) + body-z
        # torque (yaw drift), re-applied every control step (is_global=False ->
        # local link frame, so it stays body-relative).
        if self.cfg.lateral_bias_force > 0.0 or self.cfg.yaw_bias_torque > 0.0:
            self._ext_force[:, 0, 1] = self._lat_bias
            self._ext_torque[:, 0, 2] = self._yaw_bias
            self._robot.set_external_force_and_torque(self._ext_force, self._ext_torque, body_ids=self._robot_base_id)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        # IMU sensor data: simulated MPU6050 with 180° Z rotation.
        # ImuCfg offset rotates the sensor frame, so ang_vel_b is reported in
        # the rotated frame natively (no extra correction needed).
        imu = self.scene["imu"]
        ang_vel_b = imu.data.ang_vel_b * self._imu_negate
        proj_gravity = imu.data.projected_gravity_b * self._imu_negate
        # v0.9 gait-clock observation (4 dims, appended LAST): sin of each foot's
        # cycle phase. The policy must know the schedule to comply with the phase
        # rewards (Siekmann/WTW). Deployment: the policy node generates the same
        # clock from time (freq + offsets are constants of the contract).
        clock = torch.sin(2.0 * torch.pi * ((self._gait_phase.unsqueeze(1) + self._gait_offsets.unsqueeze(0)) % 1.0))
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    ang_vel_b,
                    proj_gravity,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    clock,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        # Observation noise for sim-to-sim robustness. In Isaac the robot trains
        # nearly level (projected_gravity_x std ~0.03), but in gazebo its forward
        # CoM tips the body ~14 deg -> grav_x reads -0.24, which is 8+ sigma OOD
        # and saturates the deployed policy into a frozen, railed stance (audited
        # live). Injecting noise here widens the baked obs-normalizer so the same
        # gazebo reading lands ~2 sigma instead of ~8, and forces the policy to
        # tolerate body tilt + joint-velocity jitter rather than memorising the
        # clean Isaac signal. Commands/prev_actions are exact (zero noise).
        if self._obs_noise_std is None:
            self._obs_noise_std = torch.tensor(
                [0.10] * 3 + [0.20] * 3 + [0.12] * 3 + [0.0] * 3 + [0.03] * 12 + [0.6] * 12 + [0.0] * 12 + [0.0] * 4,
                device=self.device,
            )
        obs = obs + torch.randn_like(obs) * self._obs_noise_std
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        imu = self.scene["imu"]
        ang_vel_b = imu.data.ang_vel_b * self._imu_negate
        proj_gravity = imu.data.projected_gravity_b * self._imu_negate

        # Linear velocity tracking — SHARP Gaussian on the xy command error.
        # sigma^2 was 0.25, which is far too lenient for the small forward
        # commands here: standing still at cmd 0.2 still scored exp(-0.04/0.25)
        # = 0.85, so the policy learned to shuffle in place (collecting the gait
        # rewards) and never translated -- the trained linvel_x distribution was
        # mean ~0, std 0.045. sigma^2=0.1 makes standing clearly sub-optimal
        # (~0.05 at the mean command) so the policy must actually move.
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # sigma^2 0.1 -> 0.04 (v0.9.1): with the slow command range (vx mean 0.06)
        # standing still only cost ~3.5% of this term, so the clock-schedule
        # "march in place" was near-optimal and fwd progress flatlined for 12k
        # iters. At 0.04 standing at the mean command costs ~9%, restoring a real
        # translation gradient while the clock keeps the feet honest.
        lin_vel_reward = torch.exp(-lin_vel_error / 0.04)
        # Forward progress — body-frame x velocity, rewarded LINEARLY (no
        # saturation at standstill, unlike the exp term). With forward-only
        # commands this guarantees moving always beats standing, breaking the
        # shuffle-in-place local optimum. Gated on a forward command.
        fwd_vel = self._robot.data.root_lin_vel_b[:, 0]
        forward_progress = torch.where(
            self._commands[:, 0] > 0.05,
            torch.clamp(fwd_vel, min=0.0, max=0.12),  # deliberate creep: no "faster pays" above 0.12
            torch.zeros_like(fwd_vel),
        )
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(ang_vel_b[:, :2]), dim=1)
        # yaw rate tracking — sharp Gaussian on the yaw-rate error, same shape as
        # lin_vel. No 1/|cmd|^2 blow-up (that term reached 13-22 in training and
        # made spinning the dominant reward).
        yaw_rate_error = torch.square(self._commands[:, 2] - ang_vel_b[:, 2])
        yaw_reward = torch.exp(-yaw_rate_error / 0.1)
        # LINEAR yaw-progress: reward yaw rate achieved in the COMMANDED
        # direction, capped at |cmd|, with no saturation -- so turning always
        # beats not turning (the exp term above is flat at large error and gave
        # the policy no reason to commit to a turn). Gated on a real yaw command.
        cmd_yaw = self._commands[:, 2]
        ach_yaw = ang_vel_b[:, 2]
        yaw_progress = torch.where(
            torch.abs(cmd_yaw) > 0.1,
            torch.clamp(ach_yaw * torch.sign(cmd_yaw), min=0.0),
            torch.zeros_like(cmd_yaw),
        )
        yaw_progress = torch.minimum(yaw_progress, torch.abs(cmd_yaw))
        # Anti-drift: when commanded ~straight, penalize any yaw rate so the gait
        # holds heading (counters the DART right-drift at the policy level).
        straight_gate = (torch.abs(cmd_yaw) < 0.1).float()
        yaw_straight_pen = torch.square(ach_yaw) * straight_gate
        # Anti-crab: when commanded ~straight laterally (|cmd_vy| ~ 0), penalize
        # body-frame y velocity so the gait holds its LINE. This is the lateral
        # analogue of yaw_straight_pen and is the direct cure for the systematic
        # PhysX->DART sideways/right drift -- the soft exp lin_vel tracking does
        # not punish a steady slip hard enough.
        lat_straight_gate = (torch.abs(self._commands[:, 1]) < 0.02).float()
        lat_straight_pen = torch.square(self._robot.data.root_lin_vel_b[:, 1]) * lat_straight_gate
        # Stand-still: when commanded to FULLY stop (vx~0, vy~0, wz~0), penalize
        # the body's forward velocity so the gait holds position on a zero command
        # instead of creeping forward -- the source of the post-goal drift in DART.
        stand_gate = (
            (torch.abs(self._commands[:, 0]) < 0.02)
            & (torch.abs(self._commands[:, 1]) < 0.02)
            & (torch.abs(self._commands[:, 2]) < 0.05)
        ).float()
        stand_still_pen = torch.square(self._robot.data.root_lin_vel_b[:, 0]) * stand_gate
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(proj_gravity[:, :2]), dim=1)

        # Anti-sprawl posture: keep the HIP joints near the compact default pose.
        # The explicit fine-tune drifted into a WIDE splayed stance (sum(dev^2)
        # ~0.41 vs the compact implicit gait's ~0.20). 62% of that sprawl energy
        # is in the four hip joints (idx 0,3,6,9), so penalize only those (leaving
        # thigh/knee freedom the swing needs). A full-12-joint penalty at -1.0 was
        # ~8x too weak vs crawl_gait and did nothing; hip-only at -9.0 competes.
        hip_ids = [0, 3, 6, 9]
        joint_deviation = torch.sum(
            torch.square(self._robot.data.joint_pos[:, hip_ids] - self._robot.data.default_joint_pos[:, hip_ids]),
            dim=1,
        )

        # Base-height maintenance: hold the body near the Isaac standing height
        # (~0.09 m). There was NO height term before, so nothing stopped the
        # Gazebo sink (0.094 -> 0.063 m) that collapses the stance into a pronk.
        # Peaked reward (~0.94 at the steady 0.085 m, so it doesn't distort the
        # working gait) that pays the policy to keep the stance legs extended.
        base_height = self._robot.data.root_pos_w[:, 2]
        # Wider sigma (0.035) so a range of standing heights is in-distribution:
        # in Gazebo the body settles lower (DART contact), and a too-peaky term
        # made z<0.06 fully out-of-distribution -> the policy pronked. Tolerate
        # 0.06-0.12 while still peaking at the 0.09 Isaac standing height.
        # 0.09 = the measured Isaac standing height of the new URDF (base settles
        # 0.075-0.093). A 0.20 target (introduced in 242a463) is unreachable for
        # this robot and silently zeroed the whole term (logged 0.0006 for a full
        # run); restored to the measured height.
        base_height_reward = torch.exp(-torch.square((base_height - 0.09) / 0.035))

        # Joint activity reward - encourage using all joints
        joint_vel_magnitude = torch.sum(torch.abs(self._robot.data.joint_vel), dim=1)
        num_joints = self._robot.data.joint_pos.shape[1]
        joint_activity = joint_vel_magnitude / num_joints

        # A) Individual feet air time reward (allow longer lift for natural gait)
        feet_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        feet_air_time = torch.clamp(feet_air_time, max=1.5)  # slow deliberate swings last longer
        feet_air_time_reward = torch.mean(feet_air_time, dim=1)

        # B) Crawl / wave gait reward + multi-swing guard. The clock schedule (E)
        # owns strict sequencing, but crawl_gait is kept as reinforcement of the
        # spider crawl style: exactly ONE sustained swing while moving forward,
        # with a long deliberate lead swing. (Runs at half its historic weight so
        # the strict, non-gameable clock terms stay dominant.)
        feet_air_c = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        n_swing = (feet_air_c > 0.06).float().sum(dim=1)  # genuinely-sustained swings
        lead_air = torch.clamp(feet_air_c.max(dim=1).values, max=0.45) / 0.45  # 0..1
        single = (n_swing == 1.0).float()  # exactly one airborne foot (else 0)
        fwd_gate = torch.clamp(fwd_vel / 0.10, 0.0, 1.0)
        crawl_gait_reward = lead_air * single * fwd_gate
        multi_swing_pen = torch.clamp(n_swing - 1.0, min=0.0)

        # C) Swing-foot CLEARANCE — the missing HEIGHT signal. Air TIME alone
        # rewarded a foot sliding 1 cm off the ground identically to a 4 cm step,
        # giving the fast low-lift shuffle. Reward each AIRBORNE foot (gated on
        # air>0.06 s, so a planted/sliding foot pays zero) for its world-frame
        # height reaching a clean clear lift.
        # NOTE (new-URDF recalibration): the foot link (arm_c_*_1) origin sits
        # near the knee, not the toe, so on the new geometry it stands at z~0.144 m
        # (planted) and a real swing lifts it to ~0.16-0.18 m -- measured from a
        # forward rollout. The OLD 0.045 m target was ~10 sigma below that, so this
        # 6.0-scale term was logging EXACTLY 0.0 (dead): the policy got NO foot-lift
        # gradient, which is why every leg dragged and one folded under (no per-foot
        # lift incentive). Target the clean ~3.5 cm lift (0.18 m, = the current
        # best-case swing height) so all four feet are paid to clear the ground.
        feet_z = self._robot.data.body_pos_w[:, self._feet_body_ids, 2]  # (N,4) world z
        swinging = (feet_air_c > 0.06).float()
        foot_clearance_reward = torch.sum(torch.exp(-torch.square((feet_z - 0.18) / 0.03)) * swinging, dim=1)

        # Foot-TIP kinematics (shared by D + E). The arm_c link origin is the KNEE
        # (~0.145 m above the contact), so link velocity is contaminated by body
        # motion. Compute the actual contact point via FK and its world velocity
        # tip_vel = v_link + w_link x (R*offset). Verified: planted tip z = 0.000.
        in_contact = (feet_air_c < 0.001).float()  # (N,4)
        foot_quat = self._robot.data.body_quat_w[:, self._feet_body_ids, :]  # (N,4,4)
        foot_linvel = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :]  # (N,4,3)
        foot_angvel = self._robot.data.body_ang_vel_w[:, self._feet_body_ids, :]  # (N,4,3)
        r_tip = quat_apply(foot_quat, self._tip_offset.expand(self.num_envs, 4, 3))  # knee->tip, world
        tip_vel_xy = (foot_linvel + torch.cross(foot_angvel, r_tip, dim=-1))[..., :2]  # (N,4,2) world

        # D) Stance-slip penalty: a PLANTED tip must not skid (grip the ground).
        foot_dragging = torch.sum(torch.norm(tip_vel_xy, dim=2) * in_contact, dim=1)

        # E) GAIT-CLOCK phase rewards (v0.9, Siekmann ICRA'21 / Walk-These-Ways).
        # Every contact-gated shaping term (v0.5-v0.8) converged to the sliding
        # equilibrium because the policy had no phase signal: nothing ever said
        # "THIS foot must be still NOW". A wave-gait clock (phase phi + per-foot
        # offsets, also fed to the policy as observations) assigns each foot
        # stance/swing windows:
        #   stance window -> reward a STILL foot TIP (kills both slip and the
        #     blade-edge rolling contact: the only way to hold the tip still while
        #     loaded is a non-rolling, tip-down leg configuration, which is inside
        #     the action range at calf ~2.0);
        #   swing window  -> reward ZERO contact force (the foot must genuinely
        #     unload and lift; dragging through contact scores nothing).
        # Full-stop commands force all-stance (clean standing).
        p_foot = (self._gait_phase.unsqueeze(1) + self._gait_offsets.unsqueeze(0)) % 1.0  # (N,4)
        in_stance_sched = torch.sigmoid((self.cfg.gait_stance_ratio - p_foot) * 60.0)  # ~1 stance, ~0 swing
        stop_cmd = (
            (torch.abs(self._commands[:, 0]) < 0.02)
            & (torch.abs(self._commands[:, 1]) < 0.02)
            & (torch.abs(self._commands[:, 2]) < 0.05)
        ).float().unsqueeze(1)
        in_stance_sched = torch.maximum(in_stance_sched, stop_cmd)
        tip_speed = torch.norm(tip_vel_xy, dim=2)  # (N,4) true contact-point speed
        gait_stance_still = torch.sum(in_stance_sched * torch.exp(-torch.square(tip_speed) / 0.02), dim=1)
        foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1)  # (N,4)
        gait_swing_unload = torch.sum((1.0 - in_stance_sched) * torch.exp(-torch.square(foot_forces) / 25.0), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "joint_deviation": joint_deviation * self.cfg.joint_deviation_reward_scale * self.step_dt,
            "base_height": base_height_reward * self.cfg.base_height_reward_scale * self.step_dt,
            "joint_activity": joint_activity * self.cfg.joint_activity_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time_reward * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "crawl_gait": crawl_gait_reward * self.cfg.crawl_gait_reward_scale * self.step_dt,
            "gait_stance_still": gait_stance_still * self.cfg.gait_stance_still_reward_scale * self.step_dt,
            "gait_swing_unload": gait_swing_unload * self.cfg.gait_swing_unload_reward_scale * self.step_dt,
            "foot_clearance": foot_clearance_reward * self.cfg.foot_clearance_reward_scale * self.step_dt,
            "multi_swing_pen": multi_swing_pen * self.cfg.multi_swing_penalty_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_reward * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "yaw_progress": yaw_progress * self.cfg.yaw_progress_reward_scale * self.step_dt,
            "yaw_straight_pen": yaw_straight_pen * self.cfg.yaw_straight_penalty_scale * self.step_dt,
            "lat_straight_pen": lat_straight_pen * self.cfg.lat_straight_penalty_scale * self.step_dt,
            "stand_still_pen": stand_still_pen * self.cfg.stand_still_penalty_scale * self.step_dt,
            "forward_progress": forward_progress * self.cfg.forward_progress_reward_scale * self.step_dt,
            "foot_dragging": foot_dragging * self.cfg.foot_dragging_penalty_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1), dim=1)[0] > 50.0, dim=1
        )
        # Collapse termination: if the body sinks below 0.04 m it has given way
        # (spawn z=0.1, steady ~0.085-0.09 are well above), making the sink that
        # degrades the Gazebo gait into a pronk strictly costly.
        died = died | (self._robot.data.root_pos_w[:, 2] < 0.03)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Reset event-stride state: assume planted (so a liftoff is recorded before
        # any touchdown -> no stale stride on the first landing after reset).
        self._prev_contact[env_ids] = True
        self._foot_liftoff_pos[env_ids] = 0.0
        # Random initial gait phase: desynchronizes envs and makes every global
        # phase shift reachable (also required for the mirror augmentation to map
        # onto reachable states).
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        # Sample new velocity commands. Forward-biased curriculum: the demo
        # drives the robot forward, so x is mostly positive (up to 0.6 m/s, which
        # brackets the 0.3 m/s demo command), with modest lateral and yaw for
        # steering. The previous symmetric +/-0.2 box gave no reason to prefer
        # forward, so the policy never committed to translating.
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids])
        # Forward command range narrowed 0.15-0.4 -> 0.1-0.3 m/s (issue #35): the
        # MG995-limited legs (velocity_limit 4.0 rad/s) cannot sustain the upper
        # 0.4 m/s, so commanding it only taught the policy to move too fast for
        # its own good. 0.3 m/s still brackets the 0.3 m/s demo command while
        # staying inside what the hardware can actually deliver.
        # Slow deliberate creep (was 0.1-0.3): the reference crawl is a slow wave,
        # and capping forward_progress at 0.12 needs the command in that range.
        # Low end lowered 0.05 -> 0.0 so some envs are commanded near-stationary:
        # the policy learns to STAND (and turn in place when yaw is commanded)
        # instead of always creeping forward -- this also fixes the deployment
        # quirk where vx=0 was out-of-distribution and the robot walked anyway.
        self._commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 0.12)
        self._commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(-0.05, 0.05)
        # WIDER yaw (+/-0.15 -> +/-0.6 -> +/-0.8): teach real, HARD left/right
        # rotation for obstacle avoidance so the policy can execute Nav2's sharp
        # in-place turns directly (the missing "hard turn" that left the robot
        # unable to escape obstacles); removes the policy-node yaw clamp workaround.
        self._commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(-0.8, 0.8)
        # Sample the per-episode SUSTAINED bias disturbance (sim-to-sim crab DR):
        # a constant body-frame lateral force + yaw torque held for the episode,
        # so the policy learns to actively hold its line/heading against a
        # DART-like directional push (the missing DR behind the residual drift).
        if self.cfg.lateral_bias_force > 0.0:
            self._lat_bias[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.lateral_bias_force, self.cfg.lateral_bias_force
            )
        if self.cfg.yaw_bias_torque > 0.0:
            self._yaw_bias[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.yaw_bias_torque, self.cfg.yaw_bias_torque
            )
        # If a fixed-command override is active, replace the freshly sampled
        # random commands for the reset envs with the user's fixed values. This
        # runs BEFORE _get_observations, so the policy never sees a stray random
        # command after a mid-episode reset (issue #40: zero vx/vy/omega must
        # keep the robot stationary).
        if self._command_override is not None:
            self._commands[env_ids] = self._command_override.to(self._commands.device)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
