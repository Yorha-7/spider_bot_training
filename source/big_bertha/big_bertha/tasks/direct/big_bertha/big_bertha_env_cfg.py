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
_EXPLICIT = os.environ.get("BB_ACTUATOR", "explicit").lower() == "explicit"
_SIM_DT = 1 / 500 if _EXPLICIT else 1 / 200
_DECIMATION = 10 if _EXPLICIT else 4

# A/B friction-floor toggle (BB_FRICTION_FLOOR=high|low, default high). "high"
# raises the foot-friction DR floor so the feet CAN grip (anti-slide); "low" keeps
# the historical 0.25 floor (what the old-URDF policies trained on). Ground μ stays
# 1.0, so effective foot μ = ground x foot-material (multiply combine).
_FRIC_HIGH = os.environ.get("BB_FRICTION_FLOOR", "high").lower() != "low"
_STATIC_FRIC_RANGE = (1.0, 2.0) if _FRIC_HIGH else (0.25, 2.0)
_DYNAMIC_FRIC_RANGE = (0.9, 1.6) if _FRIC_HIGH else (0.2, 1.6)

# CLEAN-EVAL toggle (BB_EVAL_CLEAN=1): for demo GIFs / metric evals. Training-time
# domain randomization stays active in the play scripts by default, so a 1-env
# recording can land a slippery per-episode friction draw, get shoved every 3-6 s,
# and carries the constant lateral-bias push -- all of which exaggerate gait flaws
# that are NOT the policy's fault. With this flag: friction pinned to the nominal
# 1.0, no pushes, no bias forces. Use for every recorded GIF and slip measurement.
_EVAL_CLEAN = os.environ.get("BB_EVAL_CLEAN", "0") == "1"
if _EVAL_CLEAN:
    # Pin at the TRAINING-DISTRIBUTION CENTERS, not nominal edges. Pinning
    # friction to 1.0 (the low edge) and added mass to 0 (below the always-added
    # 0-0.3 kg) put playback OUT of distribution: a policy that never balanced a
    # bare-mass base died every ~13 steps (23 resets/6 s) while the same
    # checkpoint walked reset-free with training DR on. Clean eval must mean
    # "typical conditions, no randomness", not "minimal conditions".
    _STATIC_FRIC_RANGE = (1.5, 1.5)
    _DYNAMIC_FRIC_RANGE = (1.25, 1.25)


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
            # Low end raised (was 0.25/0.2): training on near-frictionless ground
            # taught a slip-tolerant SKATING gait. Keep a wide robustness range but
            # off the ice so the foot_dragging penalty is physically satisfiable.
            "static_friction_range": _STATIC_FRIC_RANGE,
            "dynamic_friction_range": _DYNAMIC_FRIC_RANGE,
            "restitution_range": (0.0, 0.25),
            "num_buckets": 64,
        },
    )
    # Payload DR sized to the REAL robot. The URDF models 1.602 kg total
    # (base_link 0.264 kg; battery 0.50, lidar 0.15, circuit box 0.14, legs ~0.50).
    # A real assembled bot is a bit heavier than the CAD model (unmodelled wiring,
    # fasteners, connectors, actual LiPo), so add +0 to +0.3 kg -> total ~1.6-1.9 kg
    # (0-19% over the model). The old (1.0, 3.0) added up to ~190% of body mass onto
    # a 0.264 kg base link, forcing the policy to fight a phantom top-heavy load
    # every episode (part of the "struggling"); this brackets the true robot mass.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.0, 0.3) if not _EVAL_CLEAN else (0.15, 0.15),
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
    # Joint Coulomb friction + armature randomization: the Gazebo URDF joints carry
    # drag (damping/inertia) the ideal Isaac joint never had. Train across a range so
    # the realized joint drag in DART is in-distribution.
    joint_props = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="Revolute_.*"),
            # Distribution CENTERS in clean eval. Zero armature/Coulomb (the old pin) made
            # the explicit PD unstable (kp*dt^2/I blows up as reflected inertia -> 0): all
            # legs oscillated and the robot collapsed from spawn (23 resets/6 s); with the
            # centers it walks reset-free. Training samples these per-joint, so the
            # all-joints-zero corner never occurs in training.
            "friction_distribution_params": (0.0, 0.05) if not _EVAL_CLEAN else (0.025, 0.025),
            "armature_distribution_params": (0.0, 0.01) if not _EVAL_CLEAN else (0.005, 0.005),
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
    action_noise_std = 0.05 if not _EVAL_CLEAN else 0.0  # rad target noise; 0 in clean eval
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
    lateral_bias_force = 2.0 if not _EVAL_CLEAN else 0.0  # v0.8: 6->2 N right-sized; 0 in clean eval
    yaw_bias_torque = 0.15 if not _EVAL_CLEAN else 0.0  # v0.8 right-sized; 0 in clean eval
    action_space = 12
    observation_space = 52  # v0.9: 48 + 4 gait-clock dims (sin per foot, appended last)
    # v0.9 gait clock (wave crawl): cycle frequency + stance duty. 0.667 Hz =
    # 1.5 s cycle -> 0.375 s swings, a deliberate one-foot-at-a-time wave with
    # 3 feet nominally planted (stance ratio 0.75). Offsets live in the env.
    gait_frequency = 0.667
    # v1.2F: clock speeds up with commanded yaw (see _pre_physics_step), so turns
    # can be tracked by stepping instead of skidding.
    turn_clock_boost = 0.8
    gait_stance_ratio = 0.75
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

    # IMU sensor at the MPU6050 mesh centroid. Identity orientation so the
    # sensor frame matches base_link — imu.data.ang_vel_b / projected_gravity_b
    # are reported in base_link frame natively.
    @configclass
    class BigBerthaSceneCfg(InteractiveSceneCfg):
        imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=ImuCfg.OffsetCfg(
                pos=(0.063460, -0.064057, 0.072712),  # MPU6050 mesh centroid in base_link
                rot=(1.0, 0.0, 0.0, 0.0),  # identity: sensor frame = base_link
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
    z_vel_reward_scale = -1.0  # v1.2A: -0.35->-1.0, kill the 9mm body bob (Phase-0 metric)
    ang_vel_reward_scale = -0.10  # Roll/pitch RATE penalty (was -0.05): flatter body keeps the lidar level
    joint_torque_reward_scale = -1e-4  # v1.2A: 10x, energy-efficiency (MG995 current) + smoother stance
    joint_accel_reward_scale = -1e-7  # Minimal acceleration penalty
    action_rate_reward_scale = -0.05  # v1.2A: 5x, smoother targets = less jitter, kinder sim-to-real
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
        1.0  # v0.9: 4->1, the gait-clock schedule owns swing timing now (kept as a small lift bootstrap)
    )
    # Crawl style reinforcement: exactly one sustained swing while moving, long
    # deliberate lead swing. Restored at HALF the historic 8.0 so it reinforces
    # the crawl look without out-shouting the strict clock terms below (this
    # air-time version is satisfiable while sliding; the clock terms are not).
    crawl_gait_reward_scale = 4.0
    # v0.9 gait-clock phase rewards (Siekmann / Walk-These-Ways): stance window ->
    # still foot TIP (kills slip AND the blade-edge rolling); swing window ->
    # zero contact force (must truly unload).
    gait_stance_still_reward_scale = 6.0
    gait_swing_unload_reward_scale = 4.0
    # v1.2G Raibert foothold: place swing feet at velocity/yaw-shifted targets
    # (the literature-exact term; skid earns zero from it by construction).
    raibert_reward_scale = 6.0
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
    yaw_progress_reward_scale = 6.0  # v1.1.2: 3->6, help yaw beat the clock reward during turn-in-place
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
        8.0  # v0.9.1: 4->8 (still capped at 0.12). The clock run marched in place
        # for 12k iters -- schedule compliance out-earned translating. Doubling the
        # linear moving-always-pays term (plus the sharper lin_vel sigma) makes
        # walking clearly beat marching while the cap keeps the crawl deliberate.
    )
    # v0.6 anti-slide redesign (training only, no URDF/limit changes):
    # Foot-TIP stance-slip penalty (`foot_dragging` key; FK contact point, not the
    # knee link). v0.5 showed -2.0 was too weak -- the policy slid and paid (tip slip
    # stayed ~1.4-1.85x body speed). Raised to -4.0 so slide-and-pay stops winning.
    # Kept SMALL as an always-on auxiliary + a comparable metric across runs
    # (WTW keeps its slip penalty tiny too; the clock terms do the real work).
    # v1.2E: -0.5 -> -2.0, always-on planted-tip slip cost (Walk-These-Ways keeps a
    # real feet-slip penalty; ours had decayed to a metric). Attacks skid in walk AND turn.
    foot_dragging_penalty_scale = -2.0
    max_tilt_angle_deg = 40.0  # Reset threshold
