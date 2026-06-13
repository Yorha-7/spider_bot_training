"""Configuration for the Spdrbot robot.

The following configuration parameters are available:

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_USD_PATH = os.path.join(_REPO_ROOT, "assets", "usd", "big_bertha", "big_bertha.usd")

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
    actuators={
        # EXPLICIT PD actuator (was ImplicitActuatorCfg). The deployed gazebo
        # controller is a synchronous explicit effort-PD
        # (tau = clip(kp*(q_des-q) + kd*(0-qd), +/-effort_limit)); training
        # against Isaac's *implicit* (unconditionally stable) actuator left a
        # sim-to-sim gap where the Isaac walk would not reproduce in gazebo.
        # IdealPDActuator computes the identical explicit PD, so the policy now
        # learns against the same actuator dynamics it is deployed on. Requires
        # sim dt=1/500 for the explicit PD to be stable (kp*dt^2/I ~ 1.4), which
        # also matches the gazebo controller_manager update_rate of 500 Hz.
        "leg_joints": IdealPDActuatorCfg(
            joint_names_expr=[
                "Revolute_110",
                "Revolute_111",
                "Revolute_112",
                "Revolute_113",
                "Revolute_114",
                "Revolute_115",
                "Revolute_116",
                "Revolute_117",
                "Revolute_118",
                "Revolute_119",
                "Revolute_120",
                "Revolute_121",
            ],
            effort_limit=1.0,
            # velocity_limit lowered 5.55 -> 4.0 rad/s to match the real MG995
            # servos (issue #35: the policy drove its joints faster than the
            # hardware can). The MG995 is rated ~0.17-0.20 s/60deg at 4.8-6V,
            # i.e. 60deg / 0.20s = ~5.2 rad/s NO-LOAD; under the leg load it is
            # appreciably slower, so 4.0 rad/s is a realistic in-service ceiling
            # rather than the servo's unloaded top speed it can never hit on the
            # robot.
            velocity_limit=4.0,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)
