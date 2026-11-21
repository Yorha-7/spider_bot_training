# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os

import gymnasium as gym
import torch
from big_bertha_v2.foot_switches import foot_switches
from big_bertha_v2.walking_rewards import (
    diagonal_pair_terms,
    filtered_contact,
    height_band_cost,
    slip_cost,
    update_movement_curriculum,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from .big_bertha_v2_env_cfg import BigberthaEnvCfg


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
            [
                "right_front_lower_rigid_assembly",
                "left_front_lower_rigid_assembly",
                "left_rear_lower_rigid_assembly",
                "Right_Rear_lower_rigid_assembly",
            ],
            preserve_order=True,
        )
        # Same feet, Articulation body ids (ordering can differ from the contact sensor).
        self._feet_body_ids, _ = self._robot.find_bodies(
            [
                "right_front_lower_rigid_assembly",
                "left_front_lower_rigid_assembly",
                "left_rear_lower_rigid_assembly",
                "Right_Rear_lower_rigid_assembly",
            ],
            preserve_order=True,
        )
        # The lower-link origins are at the knees, not the contact points, so
        # link velocity alone is not a valid slip signal. These offsets were
        # already expressed in the lower-link frames: the former shoe joints
        # were fixed at identity, so retargeting force sensing preserves them.
        self._tip_offset = torch.tensor(cfg.foot_tip_offsets, device=self.device)
        if not 0 <= cfg.contact_off_threshold < cfg.contact_on_threshold:
            raise ValueError("Contact thresholds must satisfy 0 <= off < on")
        if cfg.foot_slip_tolerance < 0 or cfg.swing_min_duration < 0:
            raise ValueError("Slip tolerance and swing duration must be nonnegative")
        if cfg.base_height_tolerance < 0 or cfg.base_height_error_scale <= 0:
            raise ValueError("Invalid height-band configuration")
        if not math.isfinite(cfg.lin_vel_tracking_sigma) or cfg.lin_vel_tracking_sigma <= 0:
            raise ValueError("Linear velocity tracking tolerance must be finite and positive")
        if not 0.5 < cfg.diagonal_gait_stance_fraction < 1.0 or cfg.diagonal_gait_reward_scale < 0:
            raise ValueError("Invalid diagonal gait configuration")
        if not 0 <= cfg.diagonal_alignment_minimum_progress_gate <= 1 or cfg.diagonal_alignment_reward_scale < 0:
            raise ValueError("Invalid diagonal alignment configuration")
        if not (0 <= cfg.movement_penalty_initial <= 1
                and 0 <= cfg.movement_tracking_low < cfg.movement_tracking_high <= 1
                and cfg.movement_tracking_time_constant > 0):
            raise ValueError("Invalid movement penalty curriculum configuration")
        self._movement_tracking_ema = torch.zeros((), device=self.device)
        self._movement_penalty_multiplier = torch.tensor(
            cfg.movement_penalty_initial if cfg.movement_penalty_curriculum else 1.0,
            device=self.device,
        )
        # Keep the existing four-channel sine/cosine phase observation; using
        # diagonal offsets here would reduce it to only sine and minus sine.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        # Raibert: neutral foot-tip xy in body frame, captured on the first reward call.
        self._foot_neutral_b = None
        self._gait_offsets = torch.tensor([0.0, 0.5, 0.25, 0.75], device=self.device)
        # FR+RL swing together; FL+RR swing half a cycle later.
        self._diagonal_pair_offsets = torch.tensor([0.0, 0.5, 0.0, 0.5], device=self.device)
        self._die_body_ids, _ = self._contact_sensor.find_bodies(
            ["base_link", ".*_hip_rigid_assembly"]
        )
        self._tilt_cos = math.cos(math.radians(self.cfg.max_tilt_angle_deg))
        # Sustained per-episode lateral force + yaw torque (local frame), so the
        # policy must actively hold heading/line against a steady disturbance.
        self._robot_base_id, _ = self._robot.find_bodies("base_link")
        self._hip_joint_ids, _ = self._robot.find_joints(".*_hip_yaw")
        # Heading the robot should hold while no yaw rate is commanded. Latched
        # at every command change and on reset (see _sample_commands/_reset_idx).
        self._yaw_ref = torch.zeros(self.num_envs, device=self.device)
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

        # Reward state for a completed swing.  The clearance is relative to
        # each foot's last measured contact height, not a global z value.
        self._previous_foot_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._swing_active = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._swing_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._stance_tip_z = torch.zeros(self.num_envs, 4, device=self.device)
        self._swing_start_z = torch.zeros(self.num_envs, 4, device=self.device)
        self._swing_peak_z = torch.zeros(self.num_envs, 4, device=self.device)
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
                "base_height",
                "diagonal_gait",
                "diagonal_alignment",
                "swing_clearance",
                "stop_support",
                "impact_pen",
                "track_ang_vel_z_exp",
                "yaw_straight_pen",
                "lat_straight_pen",
                "stand_still_pen",
                "foot_dragging",
            ]
        }
        # Track which foot contributes to the aggregate slip penalty. These
        # diagnostics do not enter the reward or the policy observation.
        self._foot_drag_sums = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_contact_steps = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_contact_speed_sums = torch.zeros(self.num_envs, 4, device=self.device)
        self._foot_sample_steps = torch.zeros(self.num_envs, device=self.device)
        self._gait_pair_steps = torch.zeros(self.num_envs, device=self.device)
        self._gait_exact_pair_steps = torch.zeros(self.num_envs, device=self.device)
        self._gait_alignment_sums = torch.zeros(self.num_envs, device=self.device)

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
        boost = self._clock_boost(self._commands[:, 0], self._commands[:, 2])
        self._gait_phase = (self._gait_phase + self.cfg.gait_frequency * boost * self.step_dt) % 1.0
        # Keep the pre-clamp action so action_l2 (below) can charge its
        # magnitude. Clamping alone leaves over-driving free, so the actor
        # drifts past saturation into a bang-bang square wave.
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
                    self._robot.data.joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    clock,
                    foot_switches(self._contact_sensor.data.net_forces_w[:, self._feet_ids]),
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
                [0.10] * 3 + [0.20] * 3 + [0.12] * 3 + [0.0] * 3 + [0.03] * 12 + [0.6] * 12 + [0.0] * 12 + [0.0] * 8,
                device=self.device,
            )
        if os.environ.get("BB_EVAL_CLEAN", "0") != "1":
            obs = obs + torch.randn_like(obs) * self._obs_noise_std
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        imu = self.scene["imu"]
        ang_vel_b = imu.data.ang_vel_b * self._imu_negate
        proj_gravity = imu.data.projected_gravity_b * self._imu_negate

        # Command tracking is the task objective.  The bounded kernels also
        # reward zero velocity for stop and turn-in-place commands.
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.lin_vel_tracking_sigma**2)
        z_vel_error = torch.square(root_lin_vel_b[:, 2])
        turn_active = (torch.abs(self._commands[:, 2]) > 0.1).float()
        ang_vel_error = torch.sum(torch.square(ang_vel_b[:, :2]), dim=1) * (1.0 + 2.0 * turn_active)
        yaw_rate_error = torch.square(self._commands[:, 2] - ang_vel_b[:, 2])
        yaw_reward = torch.exp(-yaw_rate_error / 0.1)
        yaw_straight_pen = torch.square(ang_vel_b[:, 2]) * (torch.abs(self._commands[:, 2]) < 0.1).float()
        lat_straight_gate = (torch.abs(self._commands[:, 1]) < 0.02).float()
        lat_straight_pen = torch.square(root_lin_vel_b[:, 1]) * lat_straight_gate
        stand_gate = (
            (torch.abs(self._commands[:, 0]) < 0.02)
            & (torch.abs(self._commands[:, 1]) < 0.02)
            & (torch.abs(self._commands[:, 2]) < 0.05)
        ).float()
        stand_still_pen = (
            torch.sum(torch.square(root_lin_vel_b[:, :2]), dim=1) + torch.square(ang_vel_b[:, 2])
        ) * stand_gate
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_excess = torch.sum(torch.clamp(torch.abs(self._raw_actions) - 1.0, min=0.0), dim=1)
        flat_orientation = torch.sum(torch.square(proj_gravity[:, :2]), dim=1)
        base_height = self._robot.data.root_pos_w[:, 2]
        if self.cfg.base_height_penalty_enabled:
            base_height_reward = height_band_cost(
                base_height, self.cfg.base_height_target,
                self.cfg.base_height_tolerance, self.cfg.base_height_error_scale,
            )
        else:
            base_height_reward = torch.zeros_like(base_height)
        if self.cfg.movement_penalty_curriculum:
            self._movement_tracking_ema, self._movement_penalty_multiplier = update_movement_curriculum(
                self._movement_tracking_ema, self._movement_penalty_multiplier,
                self._commands[:, :2], root_lin_vel_b[:, :2], self.step_dt,
                self.cfg.movement_tracking_time_constant, self.cfg.movement_penalty_initial,
                self.cfg.movement_tracking_low, self.cfg.movement_tracking_high,
            )

        # Tip velocity must use the lower-link-origin velocity because the tip
        # offset is defined from that origin. body_lin_vel_w is a COM velocity.
        foot_quat = self._robot.data.body_link_quat_w[:, self._feet_body_ids, :]
        foot_linvel = self._robot.data.body_link_lin_vel_w[:, self._feet_body_ids, :]
        foot_angvel = self._robot.data.body_link_ang_vel_w[:, self._feet_body_ids, :]
        r_tip = quat_apply(foot_quat, self._tip_offset.expand(self.num_envs, 4, 3))
        tip_vel_xy = (foot_linvel + torch.cross(foot_angvel, r_tip, dim=-1))[..., :2]
        foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids], dim=-1)
        contact = filtered_contact(
            foot_forces, self._previous_foot_contact,
            self.cfg.contact_on_threshold, self.cfg.contact_off_threshold,
        )
        tip_speed = torch.norm(tip_vel_xy, dim=2)
        foot_dragging = slip_cost(tip_speed, contact, self.cfg.foot_slip_tolerance)
        contact_f = contact.float()
        self._foot_drag_sums += (
            contact_f * torch.relu(tip_speed - self.cfg.foot_slip_tolerance).square()
            * self.cfg.foot_dragging_penalty_scale * self._movement_penalty_multiplier * self.step_dt / 4.0
        )
        self._foot_contact_steps += contact_f
        self._foot_contact_speed_sums += tip_speed * contact_f
        self._foot_sample_steps += 1.0
        tip_z = (self._robot.data.body_link_pos_w[:, self._feet_body_ids, :] + r_tip)[..., 2]

        linear_command = torch.norm(self._commands[:, :2], dim=1) > 0.05
        turning_command = torch.abs(self._commands[:, 2]) > 0.1
        moving_command = linear_command | turning_command
        liftoff = self._previous_foot_contact & ~contact & moving_command.unsqueeze(1)
        active_swing = torch.where(liftoff, torch.ones_like(self._swing_active), self._swing_active)
        active_swing = active_swing & moving_command.unsqueeze(1)
        self._swing_air_time = torch.where(liftoff, 0.0, self._swing_air_time)
        self._swing_air_time = torch.where(
            active_swing & ~contact, self._swing_air_time + self.step_dt, self._swing_air_time
        )
        swing_start_z = torch.where(liftoff, self._stance_tip_z, self._swing_start_z)
        swing_peak_z = torch.where(
            liftoff,
            tip_z,
            torch.where(active_swing, torch.maximum(self._swing_peak_z, tip_z), self._swing_peak_z),
        )
        touchdown = active_swing & contact & ~self._previous_foot_contact
        valid_touchdown = touchdown & (self._swing_air_time >= self.cfg.swing_min_duration)
        clearance = swing_peak_z - swing_start_z
        clearance_error = (clearance - self.cfg.swing_clearance_target) / self.cfg.swing_clearance_sigma
        swing_clearance = torch.mean(valid_touchdown.float() * torch.exp(-torch.square(clearance_error)), dim=1)
        stop_support = stand_gate * torch.mean(contact.float() * torch.exp(-torch.square(tip_speed) / 0.005), dim=1)

        # Scale the impact threshold by the simulated mass rather than keeping
        # a constant from a previous URDF.
        robot_weight = torch.sum(self._robot.data.default_mass.to(foot_forces.device), dim=1) * 9.81
        impact_threshold = 3.0 * robot_weight
        impact_pen = torch.mean(torch.square(torch.clamp(foot_forces - impact_threshold.unsqueeze(1), min=0.0)), dim=1)

        self._stance_tip_z = torch.where(contact, tip_z, self._stance_tip_z)
        self._swing_start_z = swing_start_z
        self._swing_peak_z = swing_peak_z
        self._swing_active = active_swing & ~touchdown
        self._previous_foot_contact = contact

        diagonal_gait, diagonal_alignment, pair_window, exact_pattern, alignment = diagonal_pair_terms(
            contact, self._gait_phase, self._diagonal_pair_offsets,
            self._commands[:, :2], root_lin_vel_b[:, :2], self.cfg.diagonal_gait_stance_fraction,
            self.cfg.diagonal_alignment_minimum_progress_gate,
        )
        moving_pair = pair_window & (torch.sum(self._commands[:, :2].square(), dim=1) > 0.05**2)
        self._gait_pair_steps += moving_pair.float()
        self._gait_exact_pair_steps += (moving_pair & exact_pattern).float()
        self._gait_alignment_sums += alignment * moving_pair.float()

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": (
                action_rate * self.cfg.action_rate_reward_scale * self._movement_penalty_multiplier * self.step_dt
            ),
            "action_l2": action_excess * self.cfg.action_l2_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "base_height": base_height_reward * self.cfg.base_height_reward_scale * self.step_dt,
            "diagonal_gait": diagonal_gait * self.cfg.diagonal_gait_reward_scale * self.step_dt,
            "diagonal_alignment": diagonal_alignment * self.cfg.diagonal_alignment_reward_scale * self.step_dt,
            "swing_clearance": swing_clearance * self.cfg.swing_clearance_event_weight,
            "stop_support": stop_support * self.cfg.stop_support_reward_scale * self.step_dt,
            "impact_pen": impact_pen * self.cfg.impact_penalty_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_reward * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "yaw_straight_pen": yaw_straight_pen * self.cfg.yaw_straight_penalty_scale * self.step_dt,
            "lat_straight_pen": lat_straight_pen * self.cfg.lat_straight_penalty_scale * self.step_dt,
            "stand_still_pen": stand_still_pen * self.cfg.stand_still_penalty_scale * self.step_dt,
            "foot_dragging": (
                foot_dragging * self.cfg.foot_dragging_penalty_scale * self._movement_penalty_multiplier * self.step_dt
            ),
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _clock_boost(self, cmd_vx: torch.Tensor, cmd_yaw: torch.Tensor) -> torch.Tensor:
        """Phase cadence multiplier for a command, capped at 2.1x.

        Both terms take a magnitude: cadence scales with how FAST the robot is
        asked to go, not which way. Signed vx throttles the clock on a reverse
        command (negative boost below -0.3), which stops the robot stepping.

        This remains a single helper so every policy observation uses the same
        command-dependent phase rate.
        """
        return (
            1.0
            + self.cfg.turn_clock_boost * torch.clamp(torch.abs(cmd_yaw) / 0.4, max=1.0)
            + self.cfg.speed_clock_boost * torch.clamp(torch.abs(cmd_vx) / 0.3, max=1.0)
        ).clamp(max=2.1)

    def _sample_commands(self, env_ids: torch.Tensor):
        """Forward-biased (vx, vy, yaw) sampling with pivot, reverse and stop cells.

        vx in [0, 0.4] (servo-feasible), yaw in [-0.5, 0.5]. Cell split:
        30% pure turn-in-place, 20% reverse, 5% full stop, 45% forward.
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
        rev = (draw >= 0.30) & (draw < 0.50)
        stop = draw > 0.95
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        strong_yaw = torch.empty(n, device=self.device).uniform_(0.15, 0.4) * sign
        # A dedicated cell, not a widened uniform vx range, which would put most
        # negative draws near zero where they teach nothing. -0.15 is Nav2's
        # BackUp recovery speed, the only thing that commands reverse. vy and yaw
        # keep the forward-cell distributions, making this the exact C2 image of
        # the forward cell so the symmetry augmentation can pair the two.
        reverse_vx = torch.empty(n, device=self.device).uniform_(-0.15, -0.05)
        z = torch.zeros_like(self._commands[env_ids, 0])
        self._commands[env_ids, 0] = torch.where(
            turn | stop, z, torch.where(rev, reverse_vx, self._commands[env_ids, 0])
        )
        self._commands[env_ids, 1] = torch.where(turn | stop, z, self._commands[env_ids, 1])
        self._commands[env_ids, 2] = torch.where(turn, strong_yaw, torch.where(stop, z, self._commands[env_ids, 2]))
        if self._command_override is not None:
            self._commands[env_ids] = self._command_override.to(self._commands.device)
        # Re-latch the heading reference: holding a heading from before the
        # command changed would penalise the turn the robot was just told to do.
        q = self._robot.data.root_quat_w[env_ids]
        self._yaw_ref[env_ids] = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
        )

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
        self._previous_foot_contact[env_ids] = False
        self._swing_active[env_ids] = False
        self._swing_air_time[env_ids] = 0.0
        self._stance_tip_z[env_ids] = 0.0
        self._swing_start_z[env_ids] = 0.0
        self._swing_peak_z[env_ids] = 0.0
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
        for foot_idx, foot_name in enumerate(("FR", "FL", "RL", "RR")):
            extras[f"Foot/drag_reward_{foot_name}"] = (
                self._foot_drag_sums[env_ids, foot_idx].mean() / self.max_episode_length_s
            )
            extras[f"Foot/contact_fraction_{foot_name}"] = (
                self._foot_contact_steps[env_ids, foot_idx]
                / self._foot_sample_steps[env_ids].clamp_min(1.0)
            ).mean()
            extras[f"Foot/contact_speed_mps_{foot_name}"] = (
                self._foot_contact_speed_sums[env_ids, foot_idx]
                / self._foot_contact_steps[env_ids, foot_idx].clamp_min(1.0)
            ).mean()
        pair_steps = self._gait_pair_steps[env_ids].sum().clamp_min(1.0)
        extras["Gait/exact_pair_fraction"] = self._gait_exact_pair_steps[env_ids].sum() / pair_steps
        extras["Gait/contact_alignment"] = self._gait_alignment_sums[env_ids].sum() / pair_steps
        extras["Gait/moving_pair_steps"] = self._gait_pair_steps[env_ids].sum()
        self._foot_drag_sums[env_ids] = 0.0
        self._foot_contact_steps[env_ids] = 0.0
        self._foot_contact_speed_sums[env_ids] = 0.0
        self._foot_sample_steps[env_ids] = 0.0
        self._gait_pair_steps[env_ids] = 0.0
        self._gait_exact_pair_steps[env_ids] = 0.0
        self._gait_alignment_sums[env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Curriculum/movement_penalty_multiplier"] = self._movement_penalty_multiplier.clone()
        self.extras["log"]["Curriculum/movement_tracking_quality"] = self._movement_tracking_ema.clone()
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
