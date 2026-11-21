"""Robot configuration for the latest 12-joint Big Bertha description."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

_USD = Path(__file__).resolve().parents[4] / "assets/usd/big_bertha_v2/big_bertha_v2.usd"

BIG_BERTHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD),
        activate_contact_sensors=True,
        # Match v1's uniform silver robot material without regenerating the USD.
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Flat-foot standing pose measured from the four sole meshes, in
        # radians. Explicit names avoid relying on URDF joint order.
        pos=(0.0, 0.0, 0.108),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "right_rear_hip_yaw": 0.47,
            "left_rear_hip_yaw": 0.63,
            "right_front_hip_yaw": -0.51,
            "left_front_hip_yaw": -0.61,
            "right_rear_hip_pitch": 0.060,
            "left_rear_hip_pitch": -0.104,
            "right_front_hip_pitch": 0.098,
            "left_front_hip_pitch": -0.097,
            "right_rear_knee_pitch": -0.037,
            "left_rear_knee_pitch": -0.078,
            "right_front_knee_pitch": -0.066,
            "left_front_knee_pitch": -0.061,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "servos": DCMotorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_pitch", ".*_knee_pitch"],
            effort_limit=1.18,
            saturation_effort=1.18,
            velocity_limit=7.2,
            stiffness=10.0,
            damping=0.5,
        )
    },
)
