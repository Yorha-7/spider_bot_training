"""Configuration for the Spdrbot robot.

The following configuration parameters are available:

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_USD_PATH = os.path.join(_REPO_ROOT, "assets", "usd", "big_bertha", "big_bertha.usd")

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
#       phase-1 walker on it for sim-to-real fidelity. Needs dt 1/500.
_LEG_JOINTS = [f"Revolute_{_n}" for _n in range(110, 122)]
if os.environ.get("BB_ACTUATOR", "implicit").lower() == "explicit":
    _LEG_ACTUATOR = IdealPDActuatorCfg(
        joint_names_expr=_LEG_JOINTS,
        effort_limit=1.18,
        velocity_limit=6.54,
        stiffness=20.0,
        damping=2.0,
    )
else:
    _LEG_ACTUATOR = ImplicitActuatorCfg(
        joint_names_expr=_LEG_JOINTS,
        effort_limit_sim=1.18,
        velocity_limit_sim=6.54,
        stiffness=20.0,
        damping=2.0,
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
            "Revolute_111": 0.5,
            "Revolute_112": 0.0,
            "Revolute_113": 0.0,
            "Revolute_114": 0.5,
            "Revolute_115": 0.0,
            "Revolute_116": 0.0,
            "Revolute_117": 0.5,
            "Revolute_118": 0.0,
            "Revolute_119": 0.0,
            "Revolute_120": 0.5,
            "Revolute_121": 0.0,
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
    actuators={"leg_joints": _LEG_ACTUATOR},
)
