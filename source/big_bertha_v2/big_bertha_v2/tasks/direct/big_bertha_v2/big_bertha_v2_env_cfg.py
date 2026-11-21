# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from big_bertha_v2.assets.big_bertha_v2 import BIG_BERTHA_CFG

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

###############################################################################
## Simulation and policy timing
###############################################################################
# Physics advances at 500 Hz; the policy sends a target every 10 physics
# steps, or 50 Hz (0.02 s). The configured robot uses a DCMotor actuator.
# If either value changes, check actuator stability and the policy update rate.
_SIM_DT = 1 / 500
_DECIMATION = 10


@configclass
class EventCfg:
    """Material baseline and optional domain-randomization terms.

    Must be @configclass, not a plain @dataclass -- the EventManager only
    discovers terms declared on a configclass.
    """

    ###########################################################################
    ## Collision material
    ###########################################################################
    # One fixed material on every robot collision shape, in training and eval.
    # Ground friction is 1.0 with multiply combination. One deterministic
    # bucket prevents the stock randomizer from varying materials by shape.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.25, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    ###########################################################################
    ## Base mass and center of mass
    ###########################################################################
    # Optional added base mass is sampled at every reset. Both bounds are
    # zero, so the configured CAD mass is currently unchanged.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )
    # Optional startup center-of-mass offset accounts for differences between
    # the CAD and assembled robot. All three ranges are currently zero.
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )
    ###########################################################################
    ## External push disturbances
    ###########################################################################
    # The interval event can change base velocity to test recovery. Its zero
    # velocity ranges and long interval effectively disable pushes here.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10000.0, 10001.0),
        params={
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        },
    )
    ###########################################################################
    ## Actuator gains and joint properties
    ###########################################################################
    # Gain scales are sampled at reset for hip yaw, hip pitch, and knee pitch
    # separately. Each range is (1, 1), so gains remain at the asset defaults;
    # widen a range to train across different actuator responses.
    actuator_gains_hip = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_yaw"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    actuator_gains_thigh = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    actuator_gains_calf = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_pitch"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # Optional additions to joint friction and armature are sampled at reset.
    # Zero ranges add nothing; existing asset joint properties are retained.
    joint_props = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(hip_yaw|hip_pitch|knee_pitch)"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )


@configclass
class BigberthaEnvCfg(DirectRLEnvCfg):
    ###########################################################################
    ## Episode timing and policy actions
    ###########################################################################
    # Maximum episode duration and physics steps per policy update.
    episode_length_s = 20.0
    decimation = _DECIMATION
    # Clamp each policy action to [-1, 1], then multiply by this scale and
    # add it to the default joint position to form a target, in radians.
    action_scale = 0.35
    # Standard deviation of Gaussian noise added to position targets [rad].
    action_noise_std = 0.0

    ###########################################################################
    ## Servo and sensor variations
    ###########################################################################
    # Per-episode calibration offset range [rad] and probability of a one-step
    # target delay. Both are disabled at zero.
    joint_calib_range = 0.0
    action_delay_prob = 0.0
    # Per-episode IMU gravity and angular-velocity observation bias ranges;
    # zero disables these biases. Separate observation noise is in the env.
    imu_grav_bias = 0.0
    imu_gyro_bias = 0.0
    # Maximum magnitudes of sustained, randomly signed body-frame lateral
    # force [N] and yaw torque [N m] per episode; zero disables each one.
    lateral_bias_force = 0.0
    yaw_bias_torque = 0.0

    ###########################################################################
    ## Policy inputs and gait clock
    ###########################################################################
    # Twelve joint actions; the 56 observations include four foot switches.
    # No separate privileged state is supplied to the critic.
    action_space = 12
    observation_space = 56  # 52 original observations + four lower-leg force switches
    # Base gait frequency [cycles/s]. Four phase channels are observed, while
    # the diagonal reward uses separate offsets for its alternating pairs.
    gait_frequency = 0.667
    # Increase phase cadence with commanded yaw and forward speed; the combined
    # multiplier is capped in _clock_boost in the environment implementation.
    turn_clock_boost = 0.8
    speed_clock_boost = 1.1
    state_space = 0

    ###########################################################################
    ## Physics and ground material
    ###########################################################################
    # Multiply-combined ground friction and zero restitution define a flat,
    # non-bouncing contact surface. External forces are applied each iteration.
    sim: SimulationCfg = SimulationCfg(
        dt=_SIM_DT,
        render_interval=_DECIMATION,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
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

    ###########################################################################
    ## Contact and IMU sensors
    ###########################################################################
    # Track forces on the base and leg bodies at the physics rate. Three force
    # samples are retained for termination checks; air-time tracking is enabled.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(base_link|.*_hip_rigid_assembly|.*_lower_rigid_assembly)",
        history_length=3,
        update_period=_SIM_DT,
        track_air_time=True,
    )

    # The new IMU mounting frame is not specified in the URDF. Use the base
    # frame provisionally; the policy uses angular velocity and gravity here.
    @configclass
    class BigBerthaSceneCfg(InteractiveSceneCfg):
        imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            gravity_bias=(0.0, 0.0, 0.0),
        )

    ###########################################################################
    ## Parallel environments, robot, and events
    ###########################################################################
    # Clone the robot into 200 environments, 2 m apart. EventCfg defines the
    # startup, reset, and interval randomization hooks used by each environment.
    scene: BigBerthaSceneCfg = BigBerthaSceneCfg(num_envs=200, env_spacing=2.0, replicate_physics=True)

    events: EventCfg = EventCfg()

    # Robot geometry, initial pose, and actuator settings come from the asset.
    robot: ArticulationCfg = BIG_BERTHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    ###########################################################################
    ## Playback camera
    ###########################################################################
    # Close follow-camera for the 1-env play.py --video GIF (locked onto the
    # robot). Headless training never renders, so this has no training effect.
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.4, 1.4, 0.8),
        lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(1280, 720),
    )

    ###########################################################################
    ## Reward: velocity tracking and motion regularization
    ###########################################################################
    # Positive scales reward tracking; negative scales penalize motion or
    # effort. Except for swing clearance, these terms are multiplied by the
    # policy step duration in _get_rewards.
    lin_vel_reward_scale = 6.0
    # e-fold error magnitude in m/s (the exponential divides squared error
    # by this value squared). Broader than the former 0.10 m/s tolerance to
    # preserve useful reward differences while discovering faster motion.
    lin_vel_tracking_sigma = 0.20
    z_vel_reward_scale = -1.0  # heave penalty, damps body bob
    ang_vel_reward_scale = -0.10  # roll/pitch RATE penalty, keeps the lidar level
    joint_torque_reward_scale = -1e-4  # energy efficiency + smoother stance
    joint_accel_reward_scale = -3e-8  # penalizes squared joint accelerations
    action_rate_reward_scale = -0.05  # smoother action targets, kinder sim-to-real
    # Charges action magnitude BEYOND the [-1, 1] clamp only, so it is inert
    # until the policy starts to saturate. Without it the clamp makes
    # over-driving free and the actor drifts to |a| ~ 1e4 (bang-bang output).
    action_l2_reward_scale = -2e-3  # linear in excess; see _get_rewards
    ###########################################################################
    ## Reward: body orientation and height
    ###########################################################################
    # Height has no cost inside target +/- tolerance. Outside that band, the
    # excess error is normalized by error_scale and squared before weighting.
    flat_orientation_reward_scale = -5.0  # static tilt penalty, keeps the body lidar level
    base_height_penalty_enabled = False  # set False to disable the height cost
    base_height_reward_scale = -2.0  # cost only outside the standing-height band
    base_height_target = 0.108
    base_height_tolerance = 0.01
    base_height_error_scale = 0.02
    ###########################################################################
    ## Reward: diagonal gait and penalty curriculum
    ###########################################################################
    # Each diagonal pair swings for 30% of a cycle, separated by all-stance
    # intervals. Reward is zero without achieved planar command progress.
    diagonal_gait_stance_fraction = 0.70
    diagonal_gait_reward_scale = 3.0
    # A smaller, partial-match signal helps discover the pair before the exact
    # bonus is reachable. Standing still earns at most 20% of this small term.
    diagonal_alignment_reward_scale = 0.8
    diagonal_alignment_minimum_progress_gate = 0.2
    # Start action-rate and foot-slip costs at the initial multiplier. Ramp
    # toward 1 as the moving-command tracking-quality EMA rises from low to
    # high; the multiplier does not decrease during the run.
    movement_penalty_curriculum = False
    movement_penalty_initial = 0.25
    movement_tracking_time_constant = 10.0  # seconds, independent of num_envs
    movement_tracking_low = 0.5
    movement_tracking_high = 0.9
    ###########################################################################
    ## Reward: swing clearance, stop support, and impacts
    ###########################################################################
    # Completed swing rewards use a foot's lift above its most recent contact
    # height. This avoids treating any particular number of airborne feet as a
    # failure, while avoiding continuous payment for holding a foot in the air.
    swing_clearance_event_weight = 0.4  # per event, NOT multiplied by step_dt
    swing_min_duration = 0.10  # seconds airborne before a touchdown can earn reward
    swing_clearance_target = 0.04  # fixed completed-swing clearance target [m]
    swing_clearance_sigma = 0.02  # clearance reward width [m]
    stop_support_reward_scale = 2.0  # reward still, contacting feet at a stop
    # Quadratic cost above three times the current simulated robot weight.
    impact_penalty_scale = -2e-3
    ###########################################################################
    ## Reward: yaw tracking, drift, and standstill
    ###########################################################################
    yaw_rate_reward_scale = 3.0  # tracks the commanded yaw rate
    # Penalize yaw rate for near-straight commands. This discourages turning
    # but does not directly measure or correct accumulated heading error.
    yaw_straight_penalty_scale = -2.0
    # Anti-crab: penalize body-y velocity when commanded straight laterally,
    # the lateral analogue of yaw_straight_penalty_scale.
    lat_straight_penalty_scale = -6.0
    # Penalize planar velocity and yaw rate when commanded to fully stop.
    stand_still_penalty_scale = -6.0

    ###########################################################################
    ## Reward: foot slip and contact detection
    ###########################################################################
    # Applied only to contact-confirmed feet with a physically consistent tip
    # velocity.
    foot_dragging_penalty_scale = -6.0  # squared m/s; returned from the former linear cost
    foot_slip_tolerance = 0.05  # m/s
    # Hysteresis: require the higher force to enter contact, then the lower
    # force to remain in contact. This reduces flicker near the threshold.
    contact_on_threshold = 0.98  # N
    contact_off_threshold = 0.70  # N
    # FR, FL, RL, RR; centres of the lowest face of each named Foot_sole_TPU
    # visual mesh, transformed into the lower-link frame at the standing pose.
    # These remain sole proxies rather than measured PhysX contact points.
    foot_tip_offsets = (
        (-0.0168, -0.1444, 0.0228),
        (0.0175, 0.1444, -0.0222),
        (-0.0195, -0.1440, -0.0221),
        (0.0116, 0.1449, 0.0229),
    )

    ###########################################################################
    ## Episode termination
    ###########################################################################
    # Reset when body tilt exceeds this angle. Separate base-height and
    # non-foot contact termination conditions are defined in _get_dones.
    max_tilt_angle_deg = 40.0
