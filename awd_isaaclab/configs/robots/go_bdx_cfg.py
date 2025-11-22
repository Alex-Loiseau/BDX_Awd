"""Configuration for Go BDX robot."""

import torch
import os
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim import schemas

##
# Robot Configuration
##

# Get absolute path to USD file
# Find BDX_Awd root by looking for awd/ directory
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume this file is in BDX_Awd/awd_isaaclab/configs/robots/
# So BDX_Awd root is 3 levels up
_BDX_AWD_ROOT = os.path.normpath(os.path.join(_CURRENT_DIR, "../../.."))
_USD_PATH = os.path.normpath(os.path.join(_BDX_AWD_ROOT, "awd/data/assets/go_bdx/go_bdx.usd"))

GO_BDX_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        # USD file with ground plane included (created manually in Isaac Sim)
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Initial height from config (on ground)
        rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w, x, y, z) - neutral orientation
        joint_pos={
            # Default joint positions - will be set from asset properties
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # Hip joints (yaw, roll, pitch)
        "hip_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_(yaw|roll|pitch)"],
            effort_limit=100.0,  # From motor_efforts
            velocity_limit=30.0,  # From max_velocity
            stiffness=40.0,      # From go_bdx_props.yaml stiffness
            damping=1.5,         # From go_bdx_props.yaml damping
        ),
        # Knee joints
        "knee_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*knee"],
            effort_limit=100.0,
            velocity_limit=30.0,
            stiffness=35.0,      # From go_bdx_props.yaml stiffness
            damping=1.5,
        ),
        # Ankle joints
        "ankle_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle"],
            effort_limit=100.0,
            velocity_limit=30.0,
            stiffness=30.0,      # From go_bdx_props.yaml stiffness
            damping=1.5,
        ),
        # Neck joint
        "neck_joint": ImplicitActuatorCfg(
            joint_names_expr=["neck_pitch"],
            effort_limit=50.0,
            velocity_limit=30.0,
            stiffness=10.0,      # From go_bdx_props.yaml stiffness
            damping=1.5,
        ),
        # Head joints
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=["head_(pitch|yaw|roll)"],
            effort_limit=50.0,
            velocity_limit=30.0,
            stiffness=5.0,       # From go_bdx_props.yaml stiffness
            damping=1.5,
        ),
        # Antenna joints
        "antenna_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*antenna"],
            effort_limit=10.0,   # From motor_efforts
            velocity_limit=30.0,
            stiffness=3.0,       # From go_bdx_props.yaml stiffness
            damping=1.5,
        ),
    },
)

##
# Go BDX Specific Parameters
##

GO_BDX_PARAMS = {
    # Observations and actions (16 DOFs based on go_bdx_props.yaml)
    # Base duckling_obs: 3 (projected_gravity) + 16 (dof_pos) + 16 (dof_vel) + 16 (prev_actions) = 51
    # Task-specific modes will add their own task_obs on top of this base
    "num_observations": 51,  # Base observations only
    "num_actions": 16,

    # Physical properties
    "init_height": 0.0,
    "init_quat": [0.0, 0.0, 0.0, 1.0],  # IsaacGym format (x, y, z, w)
    "init_quat_isaaclab": [1.0, 0.0, 0.0, 0.0],  # IsaacLab format (w, x, y, z)

    # Termination heights
    "termination_height": -0.05,
    "head_termination_height": 0.3,

    # Key bodies for observations
    "key_bodies": ["left_foot", "right_foot"],
    "contact_bodies": ["left_foot", "right_foot"],

    # Gait period
    "period": 0.6,

    # Environment spacing
    "env_spacing": 1.0,

    # Command ranges for velocity commands
    "command_ranges": {
        "linear_x": [-0.3, 0.3],  # m/s
        "linear_y": [-0.3, 0.3],  # m/s
        "yaw": [-0.2, 0.2],       # rad/s
    },

    # Reward scales (from duckling_command.yaml)
    "reward_scales": {
        "lin_vel_xy": 0.5,
        "ang_vel_z": 0.25,
        "torque": -0.000025,
        "action_rate": 0.0,
        "stand_still": 0.0,
    },

    # Normalization scales
    "lin_vel_scale": 0.5,
    "ang_vel_scale": 0.25,

    # PD Control gains (from go_bdx_props.yaml - used for custom PD control)
    "p_gains": 25.0,     # Proportional gain
    "d_gains": 0.6,      # Derivative gain
    "max_effort": 23.7,  # Maximum effort
    "max_velocity": 30.0,  # Maximum velocity

    # Randomization
    "randomize_com": False,
    "com_range": None,
    "randomize_torques": False,
    "torque_multiplier_range": None,
}
