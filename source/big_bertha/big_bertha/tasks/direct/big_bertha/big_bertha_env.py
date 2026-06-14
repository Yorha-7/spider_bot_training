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
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # preserve_order=True so the returned order matches this list exactly:
        # arm_c_1=FR(idx0), arm_c_2=FL(idx1), arm_c_3=RL(idx2), arm_c_4=RR(idx3),
        # verified against the foot positions in the base frame from the URDF.
        self._feet_ids, _ = self._contact_sensor.find_bodies(
            ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"], preserve_order=True
        )
        self._die_body_ids, _ = self._contact_sensor.find_bodies(["arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"])
        # Diagonal foot pairs for the trot: {FR,RL}={0,2} swing together while
        # {FL,RR}={1,3} are in stance, then swap. (Was [[0,3],[1,2]] = same-side
        # legs, which let the policy satisfy the gait reward with a pronk.)
        self._foot_pairs = [[0, 2], [1, 3]]  # [FR+RL, FL+RR] diagonals

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
                "joint_activity",
                "feet_air_time",
                "crawl_gait",
                "track_ang_vel_z_exp",
                "forward_progress",
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

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
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
                [0.10] * 3 + [0.20] * 3 + [0.12] * 3 + [0.0] * 3 + [0.03] * 12 + [0.6] * 12 + [0.0] * 12,
                device=self.device,
            )
        obs = obs + torch.randn_like(obs) * self._obs_noise_std
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Linear velocity tracking — SHARP Gaussian on the xy command error.
        # sigma^2 was 0.25, which is far too lenient for the small forward
        # commands here: standing still at cmd 0.2 still scored exp(-0.04/0.25)
        # = 0.85, so the policy learned to shuffle in place (collecting the gait
        # rewards) and never translated -- the trained linvel_x distribution was
        # mean ~0, std 0.045. sigma^2=0.1 makes standing clearly sub-optimal
        # (~0.05 at the mean command) so the policy must actually move.
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / 0.1)
        # Forward progress — body-frame x velocity, rewarded LINEARLY (no
        # saturation at standstill, unlike the exp term). With forward-only
        # commands this guarantees moving always beats standing, breaking the
        # shuffle-in-place local optimum. Gated on a forward command.
        fwd_vel = self._robot.data.root_lin_vel_b[:, 0]
        forward_progress = torch.where(
            self._commands[:, 0] > 0.05,
            torch.clamp(fwd_vel, min=0.0, max=0.4),  # crawl is slow; don't pay for speed
            torch.zeros_like(fwd_vel),
        )
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # yaw rate tracking — sharp Gaussian on the yaw-rate error, same shape as
        # lin_vel. No 1/|cmd|^2 blow-up (that term reached 13-22 in training and
        # made spinning the dominant reward).
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_reward = torch.exp(-yaw_rate_error / 0.1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Joint activity reward - encourage using all joints
        joint_vel_magnitude = torch.sum(torch.abs(self._robot.data.joint_vel), dim=1)
        num_joints = self._robot.data.joint_pos.shape[1]
        joint_activity = joint_vel_magnitude / num_joints

        # A) Individual feet air time reward (allow longer lift for natural gait)
        feet_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        feet_air_time = torch.clamp(feet_air_time, max=1.0)
        feet_air_time_reward = torch.mean(feet_air_time, dim=1)

        # B) Crawl / wave gait reward — spider-like single-leg sequence: exactly
        # ONE foot in a sustained swing at a time while the other three stay
        # planted, cycling through all four legs. `lead_air` rewards the airborne
        # foot for a REAL swing (clamped to 0.35 s) -- this kills the degenerate
        # 1-timestep foot-flicks that an earlier (air_time > 1e-4) version let the
        # policy game in place. `single` gates it to a single airborne foot, so a
        # trot (2 up) or pronk (4 up) score ~0. n_swing only counts feet airborne
        # >30 ms so micro-bounces of the planted feet don't break the gate.
        feet_air_c = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        n_swing = (feet_air_c > 0.03).float().sum(dim=1)  # feet in a real swing
        lead_air = torch.clamp(feet_air_c.max(dim=1).values, max=0.35) / 0.35  # 0..1
        single = torch.exp(-torch.square(n_swing - 1.0) / 0.4)  # peak at exactly one
        # GATE on forward motion (issue #46): the one-at-a-time pattern only pays
        # when the body is actually TRANSLATING. Without this, the policy farmed
        # crawl_gait by lifting feet one-at-a-time IN PLACE -- 99% of the positive
        # reward was earnable standing still. fwd_gate ramps 0->1 over 0..0.15 m/s.
        fwd_gate = torch.clamp(fwd_vel / 0.15, 0.0, 1.0)
        crawl_gait_reward = lead_air * single * fwd_gate

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "joint_activity": joint_activity * self.cfg.joint_activity_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time_reward * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "crawl_gait": crawl_gait_reward * self.cfg.crawl_gait_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_reward * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "forward_progress": forward_progress * self.cfg.forward_progress_reward_scale * self.step_dt,
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
        self._commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(0.1, 0.3)
        self._commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(-0.05, 0.05)
        self._commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(-0.15, 0.15)
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
