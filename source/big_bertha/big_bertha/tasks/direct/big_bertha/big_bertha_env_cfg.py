# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# spdrbot3_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from big_bertha.assets.big_bertha import BIG_BERTHA_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Curriculum sim rate, matched to the actuator picked in big_bertha.py via the
# BB_ACTUATOR env var. Explicit IdealPD needs dt 1/500 (decimation 10) for
# stability (kp*dt^2/I ~ 1.4, = gazebo's 500 Hz); implicit runs the native 1/200
# (decimation 4), 2.5x faster. The policy stays 50 Hz either way (step_dt 0.02).
_EXPLICIT = os.environ.get("BB_ACTUATOR", "implicit").lower() == "explicit"
_SIM_DT = 1 / 500 if _EXPLICIT else 1 / 200
_DECIMATION = 10 if _EXPLICIT else 4


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
        # RESET (was startup): resample per-foot friction EVERY episode, not once.
        # A body-force DR plateaued because DART's crab is a contact-level foot
        # SLIP, not a body force -- the policy learned to lean against a force
        # that doesn't exist in DART. Re-rolling each leg's friction (incl. low,
        # slippery values) every episode forces the policy to reject a NEW
        # per-foot contact imbalance online each time (it cannot memorise a fixed
        # contact), which is the actual sim-to-sim mechanism. Range widened lower
        # for more slip variety.
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 2.0),
            "dynamic_friction_range": (0.2, 1.6),
            "restitution_range": (0.0, 0.25),
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
    # ACTUATOR-strength randomization (the key missing DR for the sim-to-sim gap):
    # the policy overfit to ONE exact kp/kd, so in Gazebo/DART -- where the realized
    # actuator is effectively softer/laggier -- the stance gives way and it pronks +
    # sinks. Re-sample kp x[0.7,1.3] and kd x[0.5,2.0] per episode so the gait must
    # hold across a RANGE of actuator response (DART's lands inside it).
    # Split into per-joint-group so each group's base kd (tuned to its reflected
    # inertia: HIP 0.19, THIGH 0.13, CALF 0.08 with kp=5) gets its own DR while
    # the scale distribution stays the same.
    actuator_gains_hip = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_(110|113|116|119)"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    actuator_gains_thigh = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_(111|114|117|120)"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    actuator_gains_calf = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_(112|115|118|121)"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # Joint Coulomb friction + armature randomization: the Gazebo URDF joints carry
    # drag (damping/inertia) the ideal Isaac joint never had. Train across a range so
    # the realized joint drag in DART is in-distribution.
    joint_props = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_.*"),
            "friction_distribution_params": (0.0, 0.05),
            "armature_distribution_params": (0.0, 0.01),
            "operation": "add",
        },
    )


@configclass
class BigberthaEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    # decimation + sim dt switch with the actuator (BB_ACTUATOR, see above):
    # implicit -> 4 @ 1/200 (native, 2.5x faster); explicit -> 10 @ 1/500 (PD
    # stability, matches gazebo 500 Hz). Policy stays 50 Hz (step_dt 0.02) either.
    decimation = _DECIMATION
    action_scale = 0.25
    action_noise_std = 0.05  # rad noise on joint targets (sim-to-sim actuator robustness)
    # SUSTAINED lateral-bias DR (sim-to-sim crab fix): a constant body-frame
    # sideways force + yaw torque, randomized per-episode in [-v, v] and held for
    # the whole episode (applied in big_bertha_env._pre_physics_step). Unlike the
    # impulsive push_robot, this is a STEADY directional disturbance -- the model
    # of DART's gait-induced crab -- so the policy must learn the lateral+yaw
    # authority to walk straight against it (which a velocity penalty alone, with
    # no such disturbance in PhysX, could not teach). Robot is ~3-6 kg, so a few N
    # is gentle-moderate. Set 0.0 to disable.
    # Raised 4.0 -> 6.0 N: 4 N cut the lateral crab ~60% (straight-demo perp drift
    # -0.40 -> -0.16 m) but a residual south crab still pinned the robot on the
    # wall in nav, so deployment cross-track steering alone reached B only
    # stochastically. 6 N pushes the policy to develop more lateral-holding
    # authority so the residual DART crab is smaller and reliably steerable to B.
    lateral_bias_force = 6.0  # N, max |constant body-y push|
    yaw_bias_torque = 0.4  # N*m, max |constant body-z (yaw) torque|
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=_SIM_DT,
        render_interval=_DECIMATION,
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
        update_period=_SIM_DT,  # match sim dt (1/200 implicit, 1/500 explicit)
        track_air_time=True,
    )

    # IMU sensor: simulates the MPU6050 mounted 180° rotated on the carrier board.
    # The sensor frame is offset relative to base_link to match the visual STL
    # centroid, and the 180° Z rotation is corrected in the env by negating X/Y
    # (mirroring the real hardware bridge pipeline).
    @configclass
    class BigBerthaSceneCfg(InteractiveSceneCfg):
        imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=ImuCfg.OffsetCfg(
                pos=(0.063460, -0.094057, 0.092712),  # MPU6050 mesh centroid in base_link
                rot=(0.0, 0.0, 0.0, 1.0),  # 180° about Z: q(w,x,y,z)
            ),
            gravity_bias=(0.0, 0.0, 0.0),
        )

    # scene
    scene: BigBerthaSceneCfg = BigBerthaSceneCfg(num_envs=200, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = BIG_BERTHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Close follow-camera for the 1-env play.py --video walking GIF (zoomed in on
    # the robot). origin_type="asset_root" locks it onto the robot so it tracks
    # the gait; eye is the camera offset (m) and lookat the target. Headless
    # training never renders, so this has zero effect on training.
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.4, 1.4, 0.8),
        lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(1280, 720),
    )

    # reward scales - REBALANCED so forward translation dominates. The previous
    # set let a spinning/flailing policy out-score a walker (yaw tracking blew up
    # to 13-22 while lin_vel sat near 0). lin_vel is now the clear primary; yaw a
    # modest steering term; joint_activity (which paid for raw joint speed ->
    # thrashing) is removed; gait shaping kept but below lin_vel so a trot serves
    # forward motion rather than the reverse.
    lin_vel_reward_scale = 2.0  # exp velocity tracking; reduced (paid for standing, issue #46)
    z_vel_reward_scale = -0.35  # Heave penalty (was -0.25): damp bounce that tilts the lidar into ghost walls
    ang_vel_reward_scale = -0.10  # Roll/pitch RATE penalty (was -0.05): flatter body keeps the lidar level
    joint_torque_reward_scale = -1e-5  # Minimal torque penalty
    joint_accel_reward_scale = -1e-7  # Minimal acceleration penalty
    action_rate_reward_scale = -0.01  # Action smoothness (stronger for a cleaner gait)
    flat_orientation_reward_scale = -2.0  # Static tilt penalty (was -1.5): keep the body lidar level
    joint_deviation_reward_scale = (
        -9.0
    )  # anti-sprawl on HIP joints only (idx 0,3,6,9); -1.0 full-joint was ~8x too weak vs crawl_gait and did nothing
    base_height_reward_scale = (
        2.0  # hold body near 0.09 m standing height; counters the Gazebo sink->pronk (no height term existed before)
    )
    joint_activity_reward_scale = -0.01  # PENALTY on mean|joint_vel| (issue #35): discourage fast joint motion
    gait_pattern_reward_scale = 2.0  # Deprecated: replaced by feet_air_time and alternating_gait
    feet_air_time_reward_scale = (
        3.0  # lift bootstrap, raised (#46 follow-up: forward slide had ~0 lift, bring back real foot clearance)
    )
    crawl_gait_reward_scale = 8.0  # one foot swings at a time (spider crawl); #46: enforce one-at-a-time pattern
    foot_clearance_reward_scale = 6.0  # reference crawl: reward airborne foot reaching ~0.045 m clearance
    multi_swing_penalty_scale = -2.0  # penalize 2+ feet airborne (trot/pronk) -> enforce 3-foot support tripod
    yaw_rate_reward_scale = (
        3.0  # 2->3: track the commanded yaw rate tightly so the gait turns onto a new path responsively
    )
    # LINEAR yaw-progress reward (the turn enabler). The exp yaw term above is
    # flat at large errors: turning 0 -> 0.05 rad/s barely raises it while it
    # costs the gait, so the policy never commits to turning (measured ~0.05
    # rad/s at cmd 0.5 in Gazebo -> Nav2 cannot steer). This term pays LINEARLY
    # for yaw rate achieved in the commanded direction (capped at |cmd|), giving
    # a non-saturating gradient to actually turn -- the same trick that broke the
    # shuffle-in-place optimum for forward_progress.
    yaw_progress_reward_scale = 3.0  # 2.0->3.0: commit harder to turns for Nav2's sharp obstacle-avoidance
    # Anti-drift: when commanded ~straight (|cmd_yaw|<0.1) penalize any yaw rate,
    # so the policy holds heading instead of curving (kills the systematic
    # right-drift at the source rather than relying on the deployment heading
    # controller alone).
    yaw_straight_penalty_scale = -2.0  # -1->-2: hold heading harder so the trail tracks the path, not weaves
    # Anti-crab: when commanded ~straight laterally (|cmd_vy|<0.02) penalize
    # body-y velocity, the lateral analogue of yaw_straight_penalty. This is the
    # direct cure for the systematic PhysX->DART sideways/right drift seen in
    # demo_straight -- it makes a steady sideways slip clearly sub-optimal.
    lat_straight_penalty_scale = -6.0  # -4->-6 with the 6 N bias: penalize residual body-y velocity harder still
    # Stand-still: penalize forward body velocity when commanded to fully stop, so
    # the gait holds position on a zero command (the post-goal drift fix, training side).
    stand_still_penalty_scale = -6.0
    forward_progress_reward_scale = (
        4.0  # reduced 10->4 + cap 0.12: stop the "faster always pays" gradient so the gait creeps deliberately
    )
    max_tilt_angle_deg = 40.0  # Reset threshold
