# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

# Sim rate follows the actuator (BB_ACTUATOR, set in big_bertha.py): explicit
# IdealPD needs dt 1/500 for stability (matches Gazebo's 500 Hz); implicit runs
# the native 1/200. Policy stays 50 Hz either way (step_dt 0.02).
_EXPLICIT = os.environ.get("BB_ACTUATOR", "explicit").lower() == "explicit"
_SIM_DT = 1 / 500 if _EXPLICIT else 1 / 200
_DECIMATION = 10 if _EXPLICIT else 4

# BB_FRICTION_FLOOR=high|low (default high): friction DR floor. Ground mu
# stays 1.0 (multiply combine). The high floor was lowered 1.0 -> 0.6: the
# policy had never seen a floor slicker than its own feet, and real plastic
# feet on smooth floor run mu ~0.4-0.7.
_FRIC_HIGH = os.environ.get("BB_FRICTION_FLOOR", "high").lower() != "low"
_STATIC_FRIC_RANGE = (0.6, 2.0) if _FRIC_HIGH else (0.25, 2.0)
_DYNAMIC_FRIC_RANGE = (0.5, 1.6) if _FRIC_HIGH else (0.2, 1.6)

# BB_EVAL_CLEAN=1: pins domain randomization at its training-distribution
# centers (not nominal edges -- the corners are out of distribution and were
# never seen in training) for demo recordings and metric evals.
_EVAL_CLEAN = os.environ.get("BB_EVAL_CLEAN", "0") == "1"
if _EVAL_CLEAN:
    _STATIC_FRIC_RANGE = (1.5, 1.5)
    _DYNAMIC_FRIC_RANGE = (1.25, 1.25)


@configclass
class EventCfg:
    """Domain randomization for sim-to-sim / sim-to-real robustness.

    Must be @configclass, not a plain @dataclass -- the EventManager only
    discovers terms declared on a configclass.
    """

    # Per-foot friction resampled every episode (not once), across a range that
    # keeps the foot_dragging penalty physically satisfiable.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": _STATIC_FRIC_RANGE,
            "dynamic_friction_range": _DYNAMIC_FRIC_RANGE,
            "restitution_range": (0.0, 0.25),
            "num_buckets": 64,
        },
    )
    # Added base mass brackets the real assembled robot's weight over the CAD
    # model. reset-mode (was startup) so each episode draws a fresh mass.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 0.3) if not _EVAL_CLEAN else (0.15, 0.15),
            "operation": "add",
        },
    )
    # COM offset on the base: lidar + battery are 26% of total mass and their
    # real mounting differs from CAD; a COM shift is what body tilt is made of.
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.015, 0.015) if not _EVAL_CLEAN else (0.0, 0.0),
                "y": (-0.015, 0.015) if not _EVAL_CLEAN else (0.0, 0.0),
                "z": (-0.01, 0.01) if not _EVAL_CLEAN else (0.0, 0.0),
            },
        },
    )
    # Random shove (linear + angular) every few seconds, so the policy learns
    # to recover instead of overfitting to an undisturbed trajectory.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0) if not _EVAL_CLEAN else (10000.0, 10001.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4) if not _EVAL_CLEAN else (0.0, 0.0),
                "y": (-0.4, 0.4) if not _EVAL_CLEAN else (0.0, 0.0),
                "roll": (-0.5, 0.5) if not _EVAL_CLEAN else (0.0, 0.0),
                "pitch": (-0.5, 0.5) if not _EVAL_CLEAN else (0.0, 0.0),
                "yaw": (-0.4, 0.4) if not _EVAL_CLEAN else (0.0, 0.0),
            }
        },
    )
    # Actuator-gain randomization per joint group (each has its own reflected
    # inertia, so a shared scale distribution still lands at different
    # effective kp/kd), so the gait holds across a range of actuator response.
    actuator_gains_hip = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_(110|113|116|119)"),
            "stiffness_distribution_params": (0.6, 1.4) if not _EVAL_CLEAN else (1.0, 1.0),
            "damping_distribution_params": (0.5, 2.0) if not _EVAL_CLEAN else (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    actuator_gains_thigh = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_(111|114|117|120)"),
            "stiffness_distribution_params": (0.6, 1.4) if not _EVAL_CLEAN else (1.0, 1.0),
            "damping_distribution_params": (0.5, 2.0) if not _EVAL_CLEAN else (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    actuator_gains_calf = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_(112|115|118|121)"),
            "stiffness_distribution_params": (0.6, 1.4) if not _EVAL_CLEAN else (1.0, 1.0),
            "damping_distribution_params": (0.5, 2.0) if not _EVAL_CLEAN else (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # Joint Coulomb friction + armature randomization models the real joint
    # drag an ideal Isaac joint never has. Pinned at distribution centers (not
    # zero) in clean eval -- zero armature makes the explicit PD unstable.
    joint_props = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_.*"),
            "friction_distribution_params": (0.0, 0.05) if not _EVAL_CLEAN else (0.025, 0.025),
            "armature_distribution_params": (0.0, 0.01) if not _EVAL_CLEAN else (0.005, 0.005),
            "operation": "add",
        },
    )


@configclass
class BigberthaEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = _DECIMATION
    action_scale = 0.25
    action_noise_std = 0.05 if not _EVAL_CLEAN else 0.0  # rad target noise; 0 in clean eval
    # Servo realism (see _pre_physics_step / _reset_idx): fixed per-episode
    # horn-calibration offset, 0/1-step command latency, and IMU biases
    # (observation-side only). All zero in clean eval.
    joint_calib_range = 0.03 if not _EVAL_CLEAN else 0.0  # rad, ~1.7 deg horn spline
    action_delay_prob = 0.5 if not _EVAL_CLEAN else 0.0  # P(one 20 ms step of latency)
    imu_grav_bias = 0.03 if not _EVAL_CLEAN else 0.0  # ~1.7 deg mount error
    imu_gyro_bias = 0.05 if not _EVAL_CLEAN else 0.0  # rad/s
    # Sustained per-episode lateral force + yaw torque (constant body-frame
    # disturbance, applied in _pre_physics_step), so the policy must actively
    # hold heading/line rather than just resist an impulsive push. Set 0 to disable.
    lateral_bias_force = 2.0 if not _EVAL_CLEAN else 0.0
    yaw_bias_torque = 0.15 if not _EVAL_CLEAN else 0.0
    action_space = 12
    observation_space = 52  # 48 base dims + 4 gait-clock dims (sin per foot, appended last)
    # Wave-gait clock: cycle frequency + stance duty ratio. 0.667 Hz = 1.5 s
    # cycle, one foot swinging at a time (offsets live in the env).
    gait_frequency = 0.667
    # Clock cadence scales up with commanded yaw/speed (see _pre_physics_step),
    # so turning and fast walking stay steppable instead of requiring skidding.
    turn_clock_boost = 0.8
    speed_clock_boost = 1.1
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
        update_period=_SIM_DT,
        track_air_time=True,
    )

    # IMU at the MPU6050 mesh centroid, identity orientation so the sensor
    # frame matches base_link (ang_vel_b / projected_gravity_b reported natively).
    @configclass
    class BigBerthaSceneCfg(InteractiveSceneCfg):
        imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=ImuCfg.OffsetCfg(
                pos=(0.063460, -0.064057, 0.072712),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            gravity_bias=(0.0, 0.0, 0.0),
        )

    # scene
    scene: BigBerthaSceneCfg = BigBerthaSceneCfg(num_envs=200, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = BIG_BERTHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Close follow-camera for the 1-env play.py --video GIF (locked onto the
    # robot). Headless training never renders, so this has no training effect.
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.4, 1.4, 0.8),
        lookat=(0.0, 0.0, 0.12),
        origin_type="asset_root",
        asset_name="robot",
        resolution=(1280, 720),
    )

    # Reward scales. Rebalanced from the measured tfevents decomposition of
    # v1.0.0 (gait shaping outweighed the task 3.8:1 and half the gait reward
    # was payable in place): task terms up, gait terms trimmed + vel-gated.
    # 3.0 -> 6.0 with a tighter sigma (env.py). At the old width a 39%
    # overshoot at cmd 0.12 cost 5.4% of this term, so overspeeding was
    # effectively free against forward_progress at 8.0.
    lin_vel_reward_scale = 6.0
    z_vel_reward_scale = -1.0  # heave penalty, damps body bob
    ang_vel_reward_scale = -0.10  # roll/pitch RATE penalty, keeps the lidar level
    joint_torque_reward_scale = -1e-4  # energy efficiency + smoother stance
    # -1e-7 was a hidden anti-speed term (measured -0.55/s, 55% of
    # forward_progress; joint accel grows quadratically with cadence).
    joint_accel_reward_scale = -3e-8
    action_rate_reward_scale = -0.05  # smoother action targets, kinder sim-to-real
    # Charges only for action magnitude BEYOND the [-1, 1] clamp, so it is
    # inert while the policy behaves and bites the moment it starts to
    # saturate. v1.1.0 had no such term: the clamp made over-driving free, the
    # actor drifted to |a| ~ 1e4, and the deployed policy became a bang-bang
    # square wave that only walks because sim actuators low-pass it.
    action_l2_reward_scale = -2e-3  # LINEAR in excess, see big_bertha_env.py
    flat_orientation_reward_scale = -5.0  # static tilt penalty, keeps the body lidar level
    # -9.0 was the largest penalty in the objective AND mis-indexed onto an
    # ankle; hips are the stride joints on this radial layout, keep it mild.
    joint_deviation_reward_scale = -3.0
    base_height_reward_scale = 2.0  # holds body near the 0.09 m standing height
    # -0.01 -> -0.002. A real tripod needs SHORT swings (duty 0.75 means each
    # foot is airborne a quarter of the cycle, not half), and short swings are
    # fast ones. At -0.01 the cheapest way to satisfy multi_swing_pen was to
    # stretch every swing instead, which is how v2.0.0 ended at duty 0.45 with
    # 15-20 mm of clearance against a 50 mm target.
    joint_activity_reward_scale = -0.002
    # Crawl-style reinforcement: exactly one sustained swing while moving, with
    # a deliberate lead swing. Half the weight of the clock terms below, which
    # are the non-gameable ones.
    crawl_gait_reward_scale = 4.0
    # Gait-clock phase rewards (Siekmann / Walk-These-Ways): stance window ->
    # still foot tip; swing window -> zero contact force.
    gait_stance_still_reward_scale = 6.0
    gait_swing_unload_reward_scale = 4.0
    # Raibert foothold: places swing feet at velocity/yaw-shifted targets; skid
    # earns zero from it by construction. Now vel-gated (was payable in place).
    raibert_reward_scale = 4.0
    # 12.0 -> 20.0. At 12.0 this pays 1.69, only the sixth largest reward,
    # against gait_stance_still at 4.23. Three runs have now shown the apex is
    # incentive-limited rather than geometry-limited: the ankles use ~0.12 rad
    # of the +-0.25 rad available, so a quarter of their authority. 20.0 puts
    # this in the same class as the stance terms it competes with.
    foot_clearance_reward_scale = 20.0  # FK-tip arc apex (~5 cm), vel-gated
    # -6.0 -> -2.0. Measured as the LARGEST penalty in the objective (-1.61 of
    # the v2.0.1 decomposition) and it never bought the tripod it was raised
    # for: duty stayed ~0.45 and n_swing ~2.2 across two full runs. What it did
    # buy was a direct tax on swing time, which is the same thing as a tax on
    # foot clearance, the one metric the hardware actually needs. Squaring it
    # (see big_bertha_env.py) keeps the 3-airborne case expensive while letting
    # the ordinary 2-airborne case stop fighting the clearance term.
    multi_swing_penalty_scale = -2.0  # penalize 2+ feet airborne, squared
    # Quadratic in newtons above 3x body weight (37.8 N). Ordinary stance is
    # ~7 N per foot and free; the measured 144 N peak costs ~0.067/step, which
    # is more than forward_progress pays at full speed.
    # -3e-4 -> -2e-3. The 175 N (straight) and 208 N (turn) peaks measured on
    # model_184600 are single-frame spikes, so at the old weight they vanished
    # into the episodic mean: impact_pen improved from -0.0127 to -0.0104
    # across the run while the actual peak got WORSE than the 144 N baseline.
    impact_penalty_scale = -2e-3
    # Squared radians of heading error while no yaw rate is commanded. The
    # measured +37 deg (0.65 rad) drift costs ~0.017/step.
    # DISABLED. This term is unlearnable by construction, not underpriced.
    # It charges square(yaw_now - yaw_ref), an accumulated ABSOLUTE heading
    # error, but absolute yaw appears nowhere in the 52-D observation:
    # projected_gravity is invariant to rotation about the gravity vector, and
    # ang_vel_b[2] is the yaw RATE, the derivative. The actor is a plain MLP
    # (256/256/128) with no recurrent state, so it cannot integrate that rate
    # into a heading either. The policy was being charged for something it can
    # neither see nor remember, which makes its gradient pure noise.
    #
    # Raising it -2.0 -> -12.0 for v2.0.1 did not make it learnable, it just
    # made it the SECOND LARGEST penalty in the objective (-1.50 measured) and
    # scaled the noise 6x. Both runs confirm it: flat at -2.0, flat at -12.0.
    #
    # Heading hold belongs in the outer loop, where absolute yaw exists and
    # there is state to integrate with. policy_controller already implements
    # it (heading_hold, a PI loop on odom/map yaw). yaw_straight_pen stays and
    # still penalises yaw RATE, which IS observable.
    heading_hold_penalty_scale = 0.0
    yaw_rate_reward_scale = 3.0  # tracks the commanded yaw rate
    # Linear yaw-progress reward: pays for yaw rate achieved in the commanded
    # direction (capped at |cmd|), non-saturating so the policy always commits
    # to turning rather than sitting in the flat region of the exp term above.
    yaw_progress_reward_scale = 6.0
    # Anti-drift: penalize any yaw rate when commanded straight, so the gait
    # holds heading instead of curving.
    yaw_straight_penalty_scale = -2.0
    # Anti-crab: penalize body-y velocity when commanded straight laterally,
    # the lateral analogue of yaw_straight_penalty_scale.
    lat_straight_penalty_scale = -6.0
    # Penalize forward velocity when commanded to fully stop, so the gait
    # holds position on a zero command.
    stand_still_penalty_scale = -6.0
    forward_progress_reward_scale = 8.0  # linear moving-always-pays term, capped at 0.40 m/s
    # Always-on planted-tip slip cost (small; the clock terms do the real work
    # of preventing slip, this is kept as a comparable metric across runs).
    foot_dragging_penalty_scale = -2.0
    max_tilt_angle_deg = 40.0  # reset threshold
