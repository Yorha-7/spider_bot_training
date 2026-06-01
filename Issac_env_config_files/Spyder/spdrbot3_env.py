# spdrbot3_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .spdrbot3_env_cfg import Spdrbot3EnvCfg


class Spdrbot3Env(DirectRLEnv):
    cfg: Spdrbot3EnvCfg

    def __init__(self, cfg: Spdrbot3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._die_body_ids, _ = self._contact_sensor.find_bodies(["arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"])

        # Foot bodies - 4 legs, arm_c is the foot tip of each leg
        foot_bodies = ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"]
        self._feet_ids, _ = self._contact_sensor.find_bodies(foot_bodies)

        # Tracking for gait patterns
        self._previous_foot_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "base_height_l2",
                "not_moving_penalty",
                "falling_penalty",
                "feet_air_time",
                "gait_symmetry",
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
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

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
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Base height tracking - reward standing up (higher = better now)
        base_height = self._robot.data.root_pos_w[:, 2]
        height_error = torch.square(base_height - self.cfg.target_base_height)
        base_height_reward = torch.exp(-height_error / 0.02)

        # Not moving penalty - penalize when commanded to move but stationary
        commanded_to_move = torch.norm(self._commands[:, :2], dim=1) > 0.1
        actual_velocity = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        not_moving = commanded_to_move & (actual_velocity < 0.05)
        not_moving_penalty = not_moving.float()

        # Falling penalty - HUGE negative reward for falling (traumatizing!)
        is_fallen = base_height < self.cfg.falling_height_threshold
        falling_penalty = is_fallen.float()

        # Feet air time - simplified
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]  # [N, 4]
        moving_mask = torch.norm(self._commands[:, :2], dim=1) > 0.1
        air_time_reward = torch.mean(last_air_time, dim=1) * moving_mask

        # Gait symmetry - simplified
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        foot_forces = net_contact_forces[:, :, self._feet_ids]
        current_foot_contacts = torch.any(foot_forces > 1.0, dim=-1)
        if current_foot_contacts.dim() > 2:
            current_foot_contacts = torch.any(current_foot_contacts, dim=-1)
        diagonal_1 = current_foot_contacts[:, 0] | current_foot_contacts[:, 2]
        diagonal_2 = current_foot_contacts[:, 1] | current_foot_contacts[:, 3]
        gait_symmetry_reward = (1.0 - torch.abs(diagonal_1.float() - diagonal_2.float())).squeeze()

        # Feet air time - re-enabled
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        if last_air_time.dim() > 2:
            last_air_time = last_air_time[:, :, 0]
        air_time_reward = torch.mean(last_air_time, dim=1).squeeze()

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "base_height_l2": base_height_reward * self.cfg.base_height_reward_scale * self.step_dt,
            "not_moving_penalty": not_moving_penalty * self.cfg.not_moving_penalty_scale * self.step_dt,
            "falling_penalty": falling_penalty * self.cfg.falling_penalty_scale * self.step_dt,
            "feet_air_time": air_time_reward * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "gait_symmetry": gait_symmetry_reward * self.cfg.gait_symmetry_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Terminate immediately on fall - traumatizing event ends fast
        base_height = self._robot.data.root_pos_w[:, 2]
        fallen = base_height < self.cfg.falling_height_threshold
        return fallen, time_out

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
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
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
