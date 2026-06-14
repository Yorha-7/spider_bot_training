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
    """Configuration for randomization.

    NOTE: must be @configclass, not a plain @dataclass -- the EventManager only
    discovers terms declared on a configclass. As a @dataclass these unannotated
    EventTerm attributes were invisible (EventManager reported 0 active terms),
    which is why domain randomization was silently disabled.
    """

    # Domain randomization so the gait is robust to the sim-to-sim contact gap
    # (it was overfit to a single fixed friction -> degenerated to a pronk in
    # gazebo/DART). Randomize friction widely + perturb the base mid-episode.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.4, 1.1),
            "restitution_range": (0.0, 0.1),
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
    # Random shove every few seconds, now including roll/pitch/yaw angular
    # velocity. The linear push keeps the trot balanced; the angular push tilts
    # the body so the policy learns to WALK THROUGH a real ~14 deg pitch -- which
    # is exactly what gazebo's forward CoM imposes and what froze the earlier
    # policies. Paired with the observation noise in the env, this closes the
    # sim-to-sim orientation gap.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.7, 0.7),
                "y": (-0.7, 0.7),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.4, 0.4),
            }
        },
    )


@configclass
class BigberthaEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    # decimation 4 @ sim dt 1/200 -> policy at 50 Hz (step_dt 0.02). Native rate
    # for the implicit actuator (the 1/500 sub-stepping was only needed for the
    # explicit PD's stability).
    decimation = 4
    action_scale = 0.25
    action_noise_std = 0.05  # rad noise on joint targets (sim-to-sim actuator robustness)
    action_space = 12
    observation_space = 48
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
        prim_path="/World/envs/env_.*/Robot/(base_link|arm_a_.*|arm_c_.*)",
        history_length=3,
        update_period=0.005,  # Matches sim dt = 1/200
        track_air_time=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=200, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = BIG_BERTHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # reward scales - REBALANCED so forward translation dominates. The previous
    # set let a spinning/flailing policy out-score a walker (yaw tracking blew up
    # to 13-22 while lin_vel sat near 0). lin_vel is now the clear primary; yaw a
    # modest steering term; joint_activity (which paid for raw joint speed ->
    # thrashing) is removed; gait shaping kept but below lin_vel so a trot serves
    # forward motion rather than the reverse.
    lin_vel_reward_scale = 2.0  # exp velocity tracking; reduced (paid for standing, issue #46)
    z_vel_reward_scale = -0.25  # Vertical velocity penalty (relaxed to allow natural bounce)
    ang_vel_reward_scale = -0.05  # Roll/pitch rate penalty (stronger: damp spin/wobble)
    joint_torque_reward_scale = -1e-5  # Minimal torque penalty
    joint_accel_reward_scale = -1e-7  # Minimal acceleration penalty
    action_rate_reward_scale = -0.01  # Action smoothness (stronger for a cleaner gait)
    flat_orientation_reward_scale = -1.5  # Tilt penalty
    joint_activity_reward_scale = -0.01  # PENALTY on mean|joint_vel| (issue #35): discourage fast joint motion
    gait_pattern_reward_scale = 1.0  # Deprecated: replaced by feet_air_time and alternating_gait
    feet_air_time_reward_scale = 1.0  # lift bootstrap; small so it isn't farmed in place (#46)
    crawl_gait_reward_scale = 5.0  # one foot swings at a time (spider crawl pattern)
    yaw_rate_reward_scale = 1.0  # bounded exp yaw tracking — secondary steering term
    forward_progress_reward_scale = 15.0  # PRIMARY objective: real forward translation (#46)
    max_tilt_angle_deg = 40.0  # Reset threshold
