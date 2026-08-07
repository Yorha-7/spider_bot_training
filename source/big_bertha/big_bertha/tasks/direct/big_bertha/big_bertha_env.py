# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

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
        # BB_FIXED_CMD="vx,vy,yaw" pins the command instead of sampling randomly,
        # for legible demo recordings.
        self._command_override = None
        _fixed_cmd = os.environ.get("BB_FIXED_CMD")
        if _fixed_cmd:
            self._command_override = torch.tensor([float(x) for x in _fixed_cmd.split(",")], device=self.device)

        # Per-dim observation noise std (lazily built on device in _get_observations)
        self._obs_noise_std = None

        # preserve_order=True matches this list to FR/FL/RL/RR per the URDF layout.
        self._feet_ids, _ = self._contact_sensor.find_bodies(
            ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"], preserve_order=True
        )
        # Same feet, Articulation body ids (ordering can differ from the contact sensor).
        self._feet_body_ids, _ = self._robot.find_bodies(
            ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"], preserve_order=True
        )
        # The arm_c link origin is at the knee, not the contact point, so link
        # velocity alone is not a valid slip signal. This offset (from the
        # collision mesh) maps link pose to the actual foot tip.
        self._tip_offset = torch.tensor([0.01, -0.145, -0.053], device=self.device)
        # Wave-gait clock: cycle phase + per-foot offsets (order = _feet_ids:
        # FR/FL/RL/RR). Offsets are 0.5 apart in pairs, which symmetry.py's
        # mirror augmentation relies on.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        # Raibert: neutral foot-tip xy in body frame, captured on the first reward call.
        self._foot_neutral_b = None
        self._gait_offsets = torch.tensor([0.0, 0.5, 0.25, 0.75], device=self.device)
        self._die_body_ids, _ = self._contact_sensor.find_bodies(
            ["base_link", "arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"]
        )
        self._tilt_cos = math.cos(math.radians(self.cfg.max_tilt_angle_deg))
        # Sustained per-episode lateral force + yaw torque (local frame), so the
        # policy must actively hold heading/line against a steady disturbance.
        self._robot_base_id, _ = self._robot.find_bodies("base_link")
        self._lat_bias = torch.zeros(self.num_envs, device=self.device)
        self._yaw_bias = torch.zeros(self.num_envs, device=self.device)
        self._ext_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._ext_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # ImuCfg offset is identity, so the sensor frame equals base_link.
        self._imu_negate = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # Servo realism (per-episode, resampled in _reset_idx): horn-spline
        # calibration offset and a 0/1-step command latency.
        self._joint_calib = torch.zeros(self.num_envs, 12, device=self.device)
        self._delayed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_targets = torch.zeros(self.num_envs, 12, device=self.device)
        # Per-episode IMU bias (observation-side only; rewards keep the true
        # signal): a ~2 deg mount error is a bias, and biases are the one
        # thing white noise never taught.
        self._imu_grav_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._imu_gyro_bias = torch.zeros(self.num_envs, 3, device=self.device)

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
                "action_l2",
                "flat_orientation_l2",
                "joint_deviation",
                "base_height",
                "joint_activity",
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
                "raibert",
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
        # Mid-episode command resampling (~every 4 s): Nav2 modulates /cmd_vel
        # continuously; one command per 20 s episode never trained that.
        resample = (self.episode_length_buf % 200) == 199
        if self._command_override is None and torch.any(resample):
            self._sample_commands(resample.nonzero(as_tuple=False).squeeze(-1))
        # Clock cadence scales up with commanded yaw and speed (capped 2.1x),
        # raising the max yaw rate / speed reachable by stepping instead of skidding.
        boost = (
            1.0
            + self.cfg.turn_clock_boost * torch.clamp(torch.abs(self._commands[:, 2]) / 0.4, max=1.0)
            + self.cfg.speed_clock_boost * torch.clamp(self._commands[:, 0] / 0.3, max=1.0)
        ).clamp(max=2.1)
        self._gait_phase = (self._gait_phase + self.cfg.gait_frequency * boost * self.step_dt) % 1.0
        # Keep the pre-clamp action so the reward can penalise its magnitude.
        # Clamping alone gives the policy no reason to stay inside [-1, 1]: once
        # the mean saturates, every further increase costs nothing and the actor
        # weights drift. v1.1.0 ended up emitting |a| ~ 1e4, which the clamp
        # turned into a pure bang-bang square wave. See action_l2 below.
        self._raw_actions = actions.clone()
        self._actions = torch.clamp(self._raw_actions, -1.0, 1.0)
        # Per-episode horn-calibration offset rides on the target, like a real
        # mis-splined servo horn.
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos + self._joint_calib
        )
        # Target noise mimics the tracking error of an explicit effort-PD (Gazebo,
        # real servos) vs Isaac's implicit actuator, so the gait stays stable
        # under imperfect tracking.
        if self.cfg.action_noise_std > 0.0:
            self._processed_actions = self._processed_actions + (
                torch.randn_like(self._processed_actions) * self.cfg.action_noise_std
            )
        # Per-episode 0/1-step command latency (20 ms at 50 Hz): the ROS
        # pipeline + servo PWM latch have at least one control period of it.
        delayed = torch.where(self._delayed_env.unsqueeze(1), self._prev_targets, self._processed_actions)
        self._prev_targets = self._processed_actions.clone()
        self._processed_actions = delayed
        # Re-apply the sustained per-episode bias every step (local frame).
        if self.cfg.lateral_bias_force > 0.0 or self.cfg.yaw_bias_torque > 0.0:
            self._ext_force[:, 0, 1] = self._lat_bias
            self._ext_torque[:, 0, 2] = self._yaw_bias
            self._robot.set_external_force_and_torque(self._ext_force, self._ext_torque, body_ids=self._robot_base_id)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        # ImuCfg offset is identity, so ang_vel_b / projected_gravity_b are
        # already reported in the base_link frame.
        imu = self.scene["imu"]
        ang_vel_b = imu.data.ang_vel_b * self._imu_negate + self._imu_gyro_bias
        proj_gravity = imu.data.projected_gravity_b * self._imu_negate + self._imu_grav_bias
        # Per-foot clock phase (sin), appended last. Deployment must reproduce
        # the same clock from time (frequency + offsets are contract constants).
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
        # Noise widens the obs-normalizer so it tolerates the larger tilt and
        # joint-velocity jitter Gazebo produces vs Isaac's clean signal.
        # Commands/prev_actions stay exact (zero noise).
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

        # Sharp Gaussian on xy velocity error: a lenient sigma lets standing
        # still score highly, so the policy shuffles/marches instead of
        # translating.
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / 0.04)
        # Linear (non-saturating) forward-velocity reward when commanded
        # forward, so moving always beats standing.
        fwd_vel = self._robot.data.root_lin_vel_b[:, 0]
        # Progress ALONG the commanded direction, not along +x. The old form
        # gated on cmd_vx > 0.05, so a reverse command scored zero here and the
        # largest positive term in the objective (scale 8.0) simply went absent,
        # leaving reverse to be learned from track_lin_vel_xy_exp (3.0) alone
        # against the full gait-shaping stack. Multiplying by sign(cmd_vx) is
        # bit-identical for forward commands and makes reverse pay the same.
        cmd_vx = self._commands[:, 0]
        along_cmd = fwd_vel * torch.sign(cmd_vx)
        forward_progress = torch.where(
            cmd_vx.abs() > 0.05,
            torch.clamp(along_cmd, min=0.0, max=0.40),
            torch.zeros_like(fwd_vel),
        )
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # 3x roll/pitch-rate weight during turns, damping the tilt spikes that
        # would otherwise disturb the lidar.
        turn_active = (torch.abs(self._commands[:, 2]) > 0.1).float()
        ang_vel_error = torch.sum(torch.square(ang_vel_b[:, :2]), dim=1) * (1.0 + 2.0 * turn_active)
        # Sharp Gaussian on yaw-rate error (no 1/err^2 blowup, which let
        # spinning dominate the reward).
        yaw_rate_error = torch.square(self._commands[:, 2] - ang_vel_b[:, 2])
        yaw_reward = torch.exp(-yaw_rate_error / 0.1)
        # Linear reward for yaw rate achieved in the commanded direction
        # (capped at |cmd|) -- non-saturating, so turning always beats not
        # turning.
        cmd_yaw = self._commands[:, 2]
        ach_yaw = ang_vel_b[:, 2]
        yaw_progress = torch.where(
            torch.abs(cmd_yaw) > 0.1,
            torch.clamp(ach_yaw * torch.sign(cmd_yaw), min=0.0),
            torch.zeros_like(cmd_yaw),
        )
        yaw_progress = torch.minimum(yaw_progress, torch.abs(cmd_yaw))
        # Penalize yaw rate when commanded straight, to hold heading.
        straight_gate = (torch.abs(cmd_yaw) < 0.1).float()
        yaw_straight_pen = torch.square(ach_yaw) * straight_gate
        # Penalize body-y velocity when commanded straight laterally (the
        # lateral analogue of yaw_straight_pen), to hold line.
        lat_straight_gate = (torch.abs(self._commands[:, 1]) < 0.02).float()
        lat_straight_pen = torch.square(self._robot.data.root_lin_vel_b[:, 1]) * lat_straight_gate
        # Penalize forward velocity when commanded to fully stop, so the gait
        # holds position instead of creeping.
        stand_gate = (
            (torch.abs(self._commands[:, 0]) < 0.02)
            & (torch.abs(self._commands[:, 1]) < 0.02)
            & (torch.abs(self._commands[:, 2]) < 0.05)
        ).float()
        stand_still_pen = torch.square(self._robot.data.root_lin_vel_b[:, 0]) * stand_gate
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # Penalty on the PRE-clamp action. action_rate above uses the clamped
        # value, so once the mean saturates it reads a constant and stops
        # penalising anything. Only the raw magnitude still carries gradient,
        # and without this term nothing stops the actor drifting to |a| ~ 1e4.
        # Excess beyond the clamp is what is charged for; staying inside
        # [-1, 1] is free, so the term shapes nothing until saturation starts.
        #
        # LINEAR in the excess, deliberately not squared. We fine-tune from a
        # checkpoint already sitting at |a| ~ 1e4, where a squared term is
        # -4.8e4 per step and would blow up the first update. Linear keeps the
        # gradient constant at the scale regardless of how far out the action
        # is, so the pull back toward the clamp is steady and bounded from any
        # starting magnitude.
        action_excess = torch.sum(torch.clamp(torch.abs(self._raw_actions) - 1.0, min=0.0), dim=1)
        flat_orientation = torch.sum(torch.square(proj_gravity[:, :2]), dim=1)

        # Anti-sprawl: penalize HIP joint deviation only, leaving thigh/knee
        # free for the swing. Joint order is TYPE-grouped (hips, thighs,
        # calves), so the hips are [0:4] -- [0,3,6,9] was a leg-grouped stride
        # that pinned 2 hips, 1 knee and 1 ankle (the ankle-asymmetry bug).
        hip_ids = [0, 1, 2, 3]
        joint_deviation = torch.sum(
            torch.square(self._robot.data.joint_pos[:, hip_ids] - self._robot.data.default_joint_pos[:, hip_ids]),
            dim=1,
        )

        # Peaked reward around the measured standing height (0.09 m), tight
        # sigma so bob is penalized; prevents the sink-to-pronk collapse a
        # missing height signal would allow.
        base_height = self._robot.data.root_pos_w[:, 2]
        base_height_reward = torch.exp(-torch.square((base_height - 0.09) / 0.02))

        # Joint activity: encourage using all joints (mean |joint_vel|).
        joint_vel_magnitude = torch.sum(torch.abs(self._robot.data.joint_vel), dim=1)
        num_joints = self._robot.data.joint_pos.shape[1]
        joint_activity = joint_vel_magnitude / num_joints

        # B) Crawl-style reinforcement: exactly one sustained swing while
        # moving, with a deliberate lead swing. Half-weighted vs the clock
        # terms below, which own the real (non-gameable) sequencing.
        feet_air_c = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        n_swing = (feet_air_c > 0.06).float().sum(dim=1)
        lead_air = torch.clamp(feet_air_c.max(dim=1).values, max=0.45) / 0.45
        single = (n_swing == 1.0).float()
        fwd_gate = torch.clamp(fwd_vel / 0.10, 0.0, 1.0)
        crawl_gait_reward = lead_air * single * fwd_gate
        multi_swing_pen = torch.clamp(n_swing - 1.0, min=0.0)

        # C) Swing-phase bookkeeping (clearance itself is computed below from
        # the FK tip, not the knee-origin link z it used to measure).
        swinging = (feet_air_c > 0.06).float()
        p_foot_c = (self._gait_phase.unsqueeze(1) + self._gait_offsets.unsqueeze(0)) % 1.0
        # Duty ratio shrinks with commanded speed (0.75 crawl -> 0.60 at 0.3
        # m/s); reused by the clock schedule below.
        stance_r = (0.75 - 0.15 * torch.clamp(self._commands[:, 0] / 0.3, max=1.0)).unsqueeze(1)
        mid_swing = (stance_r + 1.0) / 2.0
        mid_gate = torch.exp(-torch.square((p_foot_c - mid_swing) / 0.06))

        # Foot-tip world velocity via FK (link origin is the knee, not the
        # contact point). tip_vel = v_link + w_link x (R * offset).
        in_contact = (feet_air_c < 0.001).float()
        foot_quat = self._robot.data.body_quat_w[:, self._feet_body_ids, :]
        foot_linvel = self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :]
        foot_angvel = self._robot.data.body_ang_vel_w[:, self._feet_body_ids, :]
        r_tip = quat_apply(foot_quat, self._tip_offset.expand(self.num_envs, 4, 3))
        tip_vel_xy = (foot_linvel + torch.cross(foot_angvel, r_tip, dim=-1))[..., :2]

        # D) Stance-slip penalty: a planted tip must not skid.
        foot_dragging = torch.sum(torch.norm(tip_vel_xy, dim=2) * in_contact, dim=1)

        # E) Gait-clock phase rewards (Siekmann / Walk-These-Ways): stance
        # window rewards a still foot tip, swing window rewards zero contact
        # force. This gives the policy a phase signal so "this foot must be
        # still now" is learnable, unlike a purely contact-gated reward.
        p_foot = (self._gait_phase.unsqueeze(1) + self._gait_offsets.unsqueeze(0)) % 1.0
        in_stance_sched = torch.sigmoid((stance_r - p_foot) * 60.0)
        stop_cmd = (
            (
                (torch.abs(self._commands[:, 0]) < 0.02)
                & (torch.abs(self._commands[:, 1]) < 0.02)
                & (torch.abs(self._commands[:, 2]) < 0.05)
            )
            .float()
            .unsqueeze(1)
        )
        in_stance_sched = torch.maximum(in_stance_sched, stop_cmd)
        tip_speed = torch.norm(tip_vel_xy, dim=2)
        # Task gate: the clock rewards pay only in proportion to
        # achieved/commanded speed (forward) or yaw rate (turn-in-place) --
        # otherwise marching/skidding in place earns the full reward without
        # doing the commanded task.
        fwd_ratio = torch.clamp(fwd_vel / torch.clamp(self._commands[:, 0], min=0.05), 0.0, 1.0)
        yaw_ratio = torch.clamp(ach_yaw * torch.sign(cmd_yaw) / torch.clamp(torch.abs(cmd_yaw), min=0.1), 0.0, 1.0)
        is_forward = self._commands[:, 0] > 0.05
        is_turn_in_place = (torch.abs(self._commands[:, 0]) < 0.05) & (torch.abs(cmd_yaw) > 0.1)
        vel_gate = torch.where(
            is_forward,
            fwd_ratio,
            torch.where(is_turn_in_place, yaw_ratio, torch.ones_like(fwd_vel)),
        ).unsqueeze(1)
        # Tight kernel so skid-speed tip motion is actually penalized (a loose
        # kernel lets skidding score almost as well as planting).
        gait_stance_still = torch.sum(vel_gate * in_stance_sched * torch.exp(-torch.square(tip_speed) / 0.005), dim=1)
        foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1)
        gait_swing_unload = torch.sum(
            vel_gate * (1.0 - in_stance_sched) * torch.exp(-torch.square(foot_forces) / 25.0), dim=1
        )

        # F) Raibert foothold: desired placement = neutral position + half
        # stance time * (v_cmd + yaw x r). Skidding scores ~0 here; only
        # stepping to the target pays. Also shapes forward stride placement,
        # attacking fwd slip and front/back duty with the same term.
        tip_pos3 = self._robot.data.body_pos_w[:, self._feet_body_ids, :] + r_tip
        root_q = self._robot.data.root_quat_w
        rel = tip_pos3 - self._robot.data.root_pos_w[:, None, :]
        tips_b = quat_apply_inverse(root_q.unsqueeze(1).expand(-1, 4, -1), rel)[..., :2]
        if self._foot_neutral_b is None:
            self._foot_neutral_b = tips_b.mean(dim=0).detach().clone()
        # f_eff must mirror the real clock in _pre_physics_step exactly
        # (speed term + 2.1 cap); omitting the speed term overstated stride
        # ~2x at vx=0.3 and pinned the foothold target at the clamp.
        f_eff = self.cfg.gait_frequency * (
            1.0
            + self.cfg.turn_clock_boost * torch.clamp(torch.abs(cmd_yaw) / 0.4, max=1.0)
            + self.cfg.speed_clock_boost * torch.clamp(self._commands[:, 0] / 0.3, max=1.0)
        ).clamp(max=2.1)
        k_r = (0.5 * stance_r.squeeze(1) / f_eff).view(-1, 1, 1)
        neu = self._foot_neutral_b.unsqueeze(0)
        tang = torch.stack([-neu[..., 1], neu[..., 0]], dim=-1)
        off = k_r * (self._commands[:, :2].unsqueeze(1) + cmd_yaw.view(-1, 1, 1) * tang)
        p_des = neu + torch.clamp(off, -0.09, 0.09)
        err2 = torch.sum(torch.square(tips_b - p_des), dim=-1)
        # vel_gate: no foothold pay for marching in place.
        raibert = torch.sum(vel_gate * torch.exp(-err2 / 0.01) * swinging, dim=1)

        # Swing-foot clearance at the FK TIP (link origin is the knee): reward
        # a real ~5 cm arc apex, peaked at mid-swing, gated like the clock
        # terms so it is not payable with zero translation.
        tip_z = tip_pos3[..., 2]
        foot_clearance_reward = torch.sum(
            vel_gate * torch.exp(-torch.square((tip_z - 0.05) / 0.02)) * swinging * (0.3 + 0.7 * mid_gate), dim=1
        )

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_l2": action_excess * self.cfg.action_l2_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "joint_deviation": joint_deviation * self.cfg.joint_deviation_reward_scale * self.step_dt,
            "base_height": base_height_reward * self.cfg.base_height_reward_scale * self.step_dt,
            "joint_activity": joint_activity * self.cfg.joint_activity_reward_scale * self.step_dt,
            "crawl_gait": crawl_gait_reward * self.cfg.crawl_gait_reward_scale * self.step_dt,
            "gait_stance_still": gait_stance_still * self.cfg.gait_stance_still_reward_scale * self.step_dt,
            "gait_swing_unload": gait_swing_unload * self.cfg.gait_swing_unload_reward_scale * self.step_dt,
            "raibert": raibert * self.cfg.raibert_reward_scale * self.step_dt,
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

    def _sample_commands(self, env_ids: torch.Tensor):
        """Forward-biased (vx, vy, yaw) sampling with pivot, reverse and stop cells.

        vx in [0, 0.4] (servo-feasible), yaw in [-0.5, 0.5]. Cell split:
        30% pure turn-in-place, 12% reverse, 5% full stop, 53% forward.
        Nav2 idles between goals and reverses to recover, so both regimes are
        sampled explicitly rather than left to the tails of a uniform draw.
        """
        n = len(env_ids)
        self._commands[env_ids] = 0.0
        self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(0.0, 0.40)
        self._commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(-0.05, 0.05)
        self._commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
        draw = torch.rand(n, device=self.device)
        turn = draw < 0.30
        rev = (draw >= 0.30) & (draw < 0.42)
        stop = draw > 0.95
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        strong_yaw = torch.empty(n, device=self.device).uniform_(0.15, 0.4) * sign
        # Reverse cell. A dedicated cell rather than widening the uniform vx
        # range: uniform over [-0.15, 0.40] puts most negative draws near zero,
        # where they teach nothing. -0.15 is Nav2's BackUp recovery speed, which
        # is the only case that actually commands reverse; faster reverse would
        # spend policy capacity on a regime nothing asks for.
        # vy and yaw keep their forward-cell distributions, so this cell is the
        # exact C2 image of the forward cell (C2 maps vx -> -vx, vy -> -vy,
        # yaw -> yaw) and the symmetry augmentation can pair the two.
        reverse_vx = torch.empty(n, device=self.device).uniform_(-0.15, -0.05)
        z = torch.zeros_like(self._commands[env_ids, 0])
        self._commands[env_ids, 0] = torch.where(
            turn | stop, z, torch.where(rev, reverse_vx, self._commands[env_ids, 0])
        )
        self._commands[env_ids, 1] = torch.where(turn | stop, z, self._commands[env_ids, 1])
        self._commands[env_ids, 2] = torch.where(turn, strong_yaw, torch.where(stop, z, self._commands[env_ids, 2]))
        if self._command_override is not None:
            self._commands[env_ids] = self._command_override.to(self._commands.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # Thigh/base strike: 10 N is below body weight (12.6 N) but far above
        # gait noise; the old 50 N (4x body weight) never fired.
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1), dim=1)[0] > 10.0, dim=1
        )
        # Collapse termination: body sunk below the standing band.
        died = died | (self._robot.data.root_pos_w[:, 2] < 0.03)
        # Tilt termination (wires the previously-unread max_tilt_angle_deg).
        died = died | (self._robot.data.projected_gravity_b[:, 2] > -self._tilt_cos)
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
        # Randomize initial phase so envs desync and all mirror states are reachable.
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        # Servo realism draws (zeroed in clean eval via the cfg values).
        self._joint_calib[env_ids] = (
            torch.empty(len(env_ids), 12, device=self.device).uniform_(-1.0, 1.0) * self.cfg.joint_calib_range
        )
        self._delayed_env[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.action_delay_prob
        self._prev_targets[env_ids] = self._robot.data.default_joint_pos[env_ids]
        self._imu_grav_bias[env_ids] = (
            torch.empty(len(env_ids), 3, device=self.device).uniform_(-1.0, 1.0) * self.cfg.imu_grav_bias
        )
        self._imu_gyro_bias[env_ids] = (
            torch.empty(len(env_ids), 3, device=self.device).uniform_(-1.0, 1.0) * self.cfg.imu_gyro_bias
        )

        self._sample_commands(env_ids)

        # Per-episode sustained lateral force + yaw torque, so the policy
        # learns authority against a steady directional disturbance.
        if self.cfg.lateral_bias_force > 0.0:
            self._lat_bias[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.lateral_bias_force, self.cfg.lateral_bias_force
            )
        if self.cfg.yaw_bias_torque > 0.0:
            self._yaw_bias[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.yaw_bias_torque, self.cfg.yaw_bias_torque
            )
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
