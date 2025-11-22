#!/usr/bin/env python3
"""Script to convert URDF files to USD format for better performance.

Usage:
    ./run_isaac_direct.sh awd_isaaclab/scripts/convert_urdf_to_usd.py --robot go_bdx
    ./run_isaac_direct.sh awd_isaaclab/scripts/convert_urdf_to_usd.py --robot mini_bdx
    ./run_isaac_direct.sh awd_isaaclab/scripts/convert_urdf_to_usd.py --all
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Parse arguments first
parser = argparse.ArgumentParser(description="Convert URDF to USD")
parser.add_argument("--robot", type=str, choices=["go_bdx", "mini_bdx", "all"],
                    default="all", help="Robot to convert")
args = parser.parse_args()

# Launch Isaac Sim (headless)
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher

launcher = AppLauncher(args_cli=["--headless"])
simulation_app = launcher.app

# Import after app is created
import time
try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim import schemas
except ImportError:
    from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
    from omni.isaac.lab.sim import schemas


def convert_robot_urdf(robot_name: str, urdf_path: str, output_dir: str):
    """Convert a robot URDF to USD format.

    Args:
        robot_name: Name of the robot (for display).
        urdf_path: Path to the URDF file.
        output_dir: Directory where USD will be saved.
    """
    print(f"\n{'='*80}")
    print(f"Converting {robot_name} URDF to USD")
    print(f"{'='*80}")
    print(f"Input URDF: {urdf_path}")
    print(f"Output dir: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Configure converter
    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name=Path(urdf_path).stem + ".usd",  # Same name as URDF
        force_usd_conversion=True,  # Always reconvert
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        # Physics properties
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
    )

    print("\n[INFO] Creating converter...")
    try:
        converter = UrdfConverter(converter_cfg)

        print("[INFO] Starting conversion (this may take a moment)...")
        # Give simulation time to initialize
        time.sleep(2)

        # Run conversion
        usd_path = converter.convert()

        print(f"\n✅ Conversion complete!")
        print(f"   USD file created at: {usd_path}")

        return usd_path

    except Exception as e:
        print(f"\n❌ Conversion failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""

    print("\n" + "="*80)
    print("URDF to USD Converter")
    print("="*80)

    robots_to_convert = []

    if args.robot == "all" or args.robot == "go_bdx":
        robots_to_convert.append({
            "name": "Go BDX",
            "urdf": os.path.join(project_root, "awd/data/assets/go_bdx/go_bdx.urdf"),
            "output_dir": os.path.join(project_root, "awd/data/assets/go_bdx"),
        })

    if args.robot == "all" or args.robot == "mini_bdx":
        robots_to_convert.append({
            "name": "Mini BDX",
            "urdf": os.path.join(project_root, "awd/data/assets/mini_bdx/urdf/bdx.urdf"),
            "output_dir": os.path.join(project_root, "awd/data/assets/mini_bdx/urdf"),
        })

    # Convert each robot
    success_count = 0
    for robot in robots_to_convert:
        if not os.path.exists(robot["urdf"]):
            print(f"\n❌ ERROR: URDF not found: {robot['urdf']}")
            continue

        result = convert_robot_urdf(robot["name"], robot["urdf"], robot["output_dir"])
        if result:
            success_count += 1

    print("\n" + "="*80)
    print(f"Conversion Summary: {success_count}/{len(robots_to_convert)} successful")
    print("="*80)

    if success_count > 0:
        print("\n📝 Next steps:")
        print("1. Open the USD file in Isaac Sim GUI")
        print("2. Add a ground plane at the appropriate height")
        print("3. Save the modified USD file")
        print("4. Update robot config to use USD instead of URDF")
    print()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Give time for any pending operations
        time.sleep(1)
        simulation_app.close()
