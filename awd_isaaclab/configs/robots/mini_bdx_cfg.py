"""Configuration for Mini BDX robot."""

import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim import schemas
from isaaclab.sim.converters import UrdfConverterCfg

##
# Robot Configuration
##

MINI_BDX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UrdfFileCfg(
        # Utilise URDF directement (USD disponible après conversion optionnelle)
        asset_path="awd/data/assets/mini_bdx/urdf/bdx.urdf",
        activate_contact_sensors=True,
        fix_base=False,  # Robot mobile
        rigid_props=schemas.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=schemas.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive_props=schemas.JointDrivePropertiesCfg(
            # Valeurs par défaut - seront écrasées par les actuateurs
            drive_type="force",
            stiffness=50.0,
            damping=1.0,
        ),
        # Configuration pour la conversion URDF (nécessaire pour joint_drive)
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=50.0,  # Sera écrasé par actuators
                damping=1.0,  # Sera écrasé par actuators
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),  # Initial height from config
        rot=(1.0, 0.0, -0.08, 0.0),  # Quaternion (w, x, y, z) - converted from [0, -0.08, 0, 1]
        joint_pos={
            # Default joint positions - will be set from asset properties
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=None,  # Will be read from URDF
            velocity_limit=None,  # Will be read from URDF
            stiffness=None,  # Will be set from asset properties (p_gains)
            damping=None,  # Will be set from asset properties (d_gains)
        ),
    },
)

##
# Mini BDX Specific Parameters
##

MINI_BDX_PARAMS = {
    # Observations and actions (12 DOFs)
    # Obs: orient(3) + ang_vel(3) + commands(3) + dof_pos(12) + dof_vel(12) + actions(12) + height(1) + lin_vel(3) + noise(3) = 52
    "num_observations": 52,
    "num_actions": 12,

    # Physical properties
    "init_height": 0.18,
    "init_quat": [0.0, -0.08, 0.0, 1.0],  # IsaacGym format (x, y, z, w)
    "init_quat_isaaclab": [1.0, 0.0, -0.08, 0.0],  # IsaacLab format (w, x, y, z)

    # Termination heights
    "termination_height": 0.03,
    "head_termination_height": 0.2,

    # Key bodies for observations
    "key_bodies": ["left_foot", "right_foot"],
    "contact_bodies": ["left_foot", "right_foot"],

    # Gait period
    "period": 0.432,

    # Environment spacing
    "env_spacing": 2.0,

    # Command ranges for velocity commands
    "command_ranges": {
        "linear_x": [-0.13, 0.13],  # m/s
        "linear_y": [-0.1, 0.1],    # m/s
        "yaw": [-0.5, 0.5],          # rad/s
    },

    # Reward scales (from duckling_command.yaml)
    "reward_scales": {
        "lin_vel_xy": 0.5,
        "ang_vel_z": 0.25,
        "torque": -0.000025,
        "action_rate": -1.0,
        "stand_still": 0.0,
    },

    # Normalization scales
    "lin_vel_scale": 0.5,
    "ang_vel_scale": 0.25,

    # Randomization
    "randomize_com": False,
    "com_range": [-0.1, 0.1],
    "randomize_torques": False,
    "torque_multiplier_range": [0.85, 1.15],
}
