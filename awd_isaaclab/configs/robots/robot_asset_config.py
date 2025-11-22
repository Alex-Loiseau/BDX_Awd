"""Flexible robot asset configuration supporting both URDF and USD formats."""

import os
from typing import Literal
from isaaclab.sim.spawners.from_files import UrdfFileCfg, UsdFileCfg
from isaaclab.sim import schemas
from isaaclab.sim.converters import UrdfConverterCfg


def create_robot_spawn_cfg(
    urdf_path: str,
    usd_path: str | None = None,
    asset_format: Literal["urdf", "usd", "auto"] = "auto",
    activate_contact_sensors: bool = True,
    fix_base: bool = False,
    merge_fixed_joints: bool = False,
):
    """Create robot spawn configuration supporting both URDF and USD.

    Args:
        urdf_path: Path to URDF file.
        usd_path: Path to USD file (optional, will be auto-generated from URDF if not provided).
        asset_format: Format to use ("urdf", "usd", or "auto"). If "auto", will use USD if it exists,
                     otherwise URDF.
        activate_contact_sensors: Whether to activate contact sensors.
        fix_base: Whether to fix the robot base.
        merge_fixed_joints: Whether to merge fixed joints.

    Returns:
        UrdfFileCfg or UsdFileCfg depending on the selected format.
    """
    # Determine which format to use
    if asset_format == "auto":
        # Auto-detect: use USD if it exists, otherwise URDF
        if usd_path and os.path.exists(usd_path):
            use_usd = True
        else:
            use_usd = False
    elif asset_format == "usd":
        use_usd = True
        if not usd_path:
            # Generate USD path from URDF path
            usd_path = urdf_path.replace(".urdf", ".usd")
    else:  # urdf
        use_usd = False

    # Common physics properties
    rigid_props = schemas.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=10.0,
    )

    articulation_props = schemas.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
    )

    if use_usd:
        # Use USD format (faster loading, better performance)
        print(f"[RobotConfig] Using USD format: {usd_path}")
        return UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=activate_contact_sensors,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    else:
        # Use URDF format (more flexible, slower)
        print(f"[RobotConfig] Using URDF format: {urdf_path}")
        return UrdfFileCfg(
            asset_path=urdf_path,
            activate_contact_sensors=activate_contact_sensors,
            fix_base=fix_base,
            merge_fixed_joints=merge_fixed_joints,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
            joint_drive_props=schemas.JointDrivePropertiesCfg(
                drive_type="force",
                stiffness=50.0,
                damping=1.0,
            ),
            # URDF converter configuration
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=50.0,
                    damping=1.0,
                ),
            ),
        )
