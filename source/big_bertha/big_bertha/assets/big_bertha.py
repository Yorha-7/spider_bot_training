"""Configuration for the Spdrbot robot.

The following configuration parameters are available:

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_USD_PATH = os.path.join(_REPO_ROOT, "assets", "usd", "big_bertha", "big_bertha.usd")

# --- Per-joint-group PD gains (tuned to reflected inertia from URDF) --------
# Each joint group has different reflected inertia, so the same kp/kd would
# give very different damping ratios.  for current experimentation, only room 
# for improvement is ceated.
#   HIP    (110,113,116,119): I_eff ≈ 0.0014-0.0026 kg·m² → kp=5, kd=0.19
#   THIGH  (111,114,117,120): I_eff ≈ 0.00115 kg·m²       → kp=5, kd=0.13
#   CALF   (112,115,118,121): I_eff ≈ 0.00048 kg·m²       → kp=5, kd=0.08
# The old kp=20, kd=2 gave ζ=4-10 and saturated the motor at just 0.059 rad
# of position error (bang-bang behavior).
# --- Curriculum actuator switch (no manual edits between phases) ------------
# Set by the BB_ACTUATOR env var; big_bertha_env_cfg.py reads the same var to
# pick the matching sim dt. Both model 12x TowerPro MG995 @6.6V: 1.18 N*m stall
# (12 kgf*cm), 6.54 rad/s no-load (0.16 s/60deg).
#   "implicit" (default, curriculum phase 1): PD solved in the physics engine ->
#       leg-lifting is easy to discover, so the policy quickly learns to WALK
#       FORWARD under the forward-gated reward (issue #46). Native dt 1/200.
#   "explicit" (phase 2): IdealPD computing the same explicit effort-PD as
#       gazebo's JointEffortPdController / the real MG995 motors
#       (tau = clip(kp*(q_des-q) + kd*(0-qd), +/-effort_limit)); fine-tune the
#       phase-1 walker on it for sim-to-sim fidelity. Needs dt 1/500.
_HIP_JOINTS = ["Revolute_110", "Revolute_113", "Revolute_116", "Revolute_119"]
_THIGH_JOINTS = ["Revolute_111", "Revolute_114", "Revolute_117", "Revolute_120"]
_CALF_JOINTS = ["Revolute_112", "Revolute_115", "Revolute_118", "Revolute_121"]

if os.environ.get("BB_ACTUATOR", "implicit").lower() == "explicit":
    _HIP_ACT = IdealPDActuatorCfg(
        joint_names_expr=_HIP_JOINTS,
        effort_limit=1.18,
        velocity_limit=6.54,
        stiffness=20.0,
        damping=2,
    )
    _THIGH_ACT = IdealPDActuatorCfg(
        joint_names_expr=_THIGH_JOINTS,
        effort_limit=1.18,
        velocity_limit=6.54,
        stiffness=20,
        damping=2,
    )
    _CALF_ACT = IdealPDActuatorCfg(
        joint_names_expr=_CALF_JOINTS,
        effort_limit=1.18,
        velocity_limit=6.54,
        stiffness=20,
        damping=2,
    )
else:
    _HIP_ACT = ImplicitActuatorCfg(
        joint_names_expr=_HIP_JOINTS,
        effort_limit_sim=1.18,
        velocity_limit_sim=6.54,
        stiffness=20,
        damping=2,
    )
    _THIGH_ACT = ImplicitActuatorCfg(
        joint_names_expr=_THIGH_JOINTS,
        effort_limit_sim=1.18,
        velocity_limit_sim=6.54,
        stiffness=20,
        damping=2,
    )
    _CALF_ACT = ImplicitActuatorCfg(
        joint_names_expr=_CALF_JOINTS, effort_limit_sim=1.18, velocity_limit_sim=6.54, stiffness=20, damping=2
    )

BIG_BERTHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "Revolute_110": 0.0,
            "Revolute_111": -0.32,
            "Revolute_112": 1.82,
            "Revolute_113": 0.0,
            "Revolute_114": -0.32,
            "Revolute_115": 1.82,
            "Revolute_116": 0.0,
            "Revolute_117": -0.32,
            "Revolute_118": 1.82,
            "Revolute_119": 0.0,
            "Revolute_120": -0.32,
            "Revolute_121": 1.82,
        },
        joint_vel={
            "Revolute_110": 0.0,
            "Revolute_111": 0.0,
            "Revolute_112": 0.0,
            "Revolute_113": 0.0,
            "Revolute_114": 0.0,
            "Revolute_115": 0.0,
            "Revolute_116": 0.0,
            "Revolute_117": 0.0,
            "Revolute_118": 0.0,
            "Revolute_119": 0.0,
            "Revolute_120": 0.0,
            "Revolute_121": 0.0,
        },
    ),
    actuators={
        "hip_joints": _HIP_ACT,
        "thigh_joints": _THIGH_ACT,
        "calf_joints": _CALF_ACT,
    },
)
