# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# spdrbot3_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from big_bertha.assets.big_bertha import BIG_BERTHA_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class BigberthaEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 30
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",  # Path to base_link in cloned envs
        history_length=5,
        update_period=0.005,  # Matches sim dt = 1/200
        track_air_time=True,  # Not needed for base
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = BIG_BERTHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # reward scales - SIMPLIFIED FOR STABILITY
    lin_vel_reward_scale = 5.0  # Velocity tracking reward
    yaw_rate_reward_scale = 1.0  # Yaw tracking reward
    z_vel_reward_scale = -2.0  # Reduced vertical velocity penalty
    ang_vel_reward_scale = -0.02  # Reduced roll/pitch penalty
    joint_torque_reward_scale = -1e-5  # Minimal torque penalty
    joint_accel_reward_scale = -1e-7  # Minimal acceleration penalty
    action_rate_reward_scale = -0.01  # Action smoothness
    flat_orientation_reward_scale = -1.5  # Reduced tilt penalty
    joint_activity_reward_scale = 0.1  # Reward for using all joints
    gait_pattern_reward_scale = 0.3  # Deprecated: replaced by feet_air_time and alternating_gait
    feet_air_time_reward_scale = 0.01  # Reward for individual foot lift
    alternating_gait_reward_scale = 0.2  # Reward for trot pattern (diagonal pairs)
    max_tilt_angle_deg = 40.0  # Reset threshold
