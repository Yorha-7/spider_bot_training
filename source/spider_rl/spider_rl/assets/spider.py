"""Configuration for the Spider robot.

The spider robot has 4 legs with 3 joints each (hip, thigh, calf).
Total: 12 actuated joints.

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_USD_PATH = os.path.join(_REPO_ROOT, "assets", "usd", "spider_rl", "spider.usd")

SPIDER_CFG = ArticulationCfg(
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
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.31,
            "FL_calf_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.31,
            "RL_calf_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.31,
            "FR_calf_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.31,
            "RR_calf_joint": 0.0,
        },
        joint_vel={
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.0,
            "FL_calf_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.0,
            "RL_calf_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.0,
            "FR_calf_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.0,
            "RR_calf_joint": 0.0,
        },
    ),
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ],
            effort_limit=20.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=1.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
"""Configuration of Spider robot using implicit actuators."""
