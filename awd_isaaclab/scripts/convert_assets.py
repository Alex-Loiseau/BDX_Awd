#!/usr/bin/env python3
"""Script to convert URDF assets to USD format for IsaacLab.

This script automates the conversion of robot URDF files to USD format,
which is preferred by IsaacLab for better performance.

IMPORTANT: This script requires IsaacLab to be installed.
Run this from Isaac Sim's Python environment:

    # Method 1: Using isaaclab.sh wrapper (recommended)
    cd /path/to/IsaacLab
    ./isaaclab.sh -p /path/to/BDX_Awd/awd_isaaclab/scripts/convert_assets.py --all

    # Method 2: Direct Python
    source /home/alexandre/Developpements/env_isaaclab/bin/activate
    python convert_assets.py --all

Usage:
    python convert_assets.py --all
    python convert_assets.py --robot mini_bdx
    python convert_assets.py --robot go_bdx
"""

import argparse
import os
import sys
from pathlib import Path


def check_isaaclab_installed() -> bool:
    """Check if IsaacLab is installed and accessible.

    Returns:
        True if IsaacLab is installed, False otherwise.
    """
    try:
        import omni.isaac.lab
        return True
    except ImportError:
        return False


def convert_urdf_to_usd(urdf_path: str, output_path: str, make_instanceable: bool = True) -> bool:
    """Convert URDF file to USD.

    Args:
        urdf_path: Path to input URDF file.
        output_path: Path to output USD file.
        make_instanceable: Whether to make the USD instanceable for better performance.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    # Check if IsaacLab is available
    if not check_isaaclab_installed():
        print("\n❌ ERROR: IsaacLab is not installed or not accessible!")
        print("\nPlease install IsaacLab first:")
        print("  cd /home/alexandre/Developpements")
        print("  git clone https://github.com/isaac-sim/IsaacLab.git")
        print("  cd IsaacLab")
        print("  ./isaaclab.sh --install")
        print("\nThen run this script using:")
        print("  ./isaaclab.sh -p /path/to/BDX_Awd/awd_isaaclab/scripts/convert_assets.py --all")
        return False

    try:
        # Import Isaac Sim converter
        from omni.isaac.lab.utils.assets import urdf_converter

        print(f"Converting: {urdf_path} -> {output_path}")

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Convert
        urdf_converter.convert_urdf_to_usd(
            urdf_path=urdf_path,
            usd_path=output_path,
            usd_as_instanceable=make_instanceable,
        )

        print(f"✓ Conversion successful: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False


def convert_robot(robot_name: str, project_root: str) -> bool:
    """Convert a specific robot's URDF to USD.

    Args:
        robot_name: Name of the robot (e.g., 'mini_bdx', 'go_bdx').
        project_root: Root directory of the project.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    # Define paths based on robot
    if robot_name == "mini_bdx":
        urdf_path = os.path.join(
            project_root, "awd/data/assets/mini_bdx/urdf/bdx.urdf"
        )
        usd_path = os.path.join(
            project_root, "awd/data/assets/mini_bdx/bdx.usd"
        )
    elif robot_name == "go_bdx":
        urdf_path = os.path.join(
            project_root, "awd/data/assets/go_bdx/go_bdx.urdf"
        )
        usd_path = os.path.join(
            project_root, "awd/data/assets/go_bdx/go_bdx.usd"
        )
    else:
        print(f"ERROR: Unknown robot: {robot_name}")
        return False

    # Check if URDF exists
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}")
        return False

    # Convert
    return convert_urdf_to_usd(urdf_path, usd_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Convert URDF assets to USD")
    parser.add_argument(
        "--robot",
        type=str,
        choices=["mini_bdx", "go_bdx"],
        help="Specific robot to convert",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all robots",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconversion even if USD already exists",
    )

    args = parser.parse_args()

    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))

    print(f"Project root: {project_root}")
    print("="*80)

    # Determine which robots to convert
    if args.all:
        robots = ["mini_bdx", "go_bdx"]
    elif args.robot:
        robots = [args.robot]
    else:
        print("ERROR: Must specify either --robot or --all")
        parser.print_help()
        sys.exit(1)

    # Convert each robot
    success_count = 0
    for robot in robots:
        print(f"\nConverting {robot}...")
        if convert_robot(robot, project_root):
            success_count += 1

    # Summary
    print("\n" + "="*80)
    print(f"Conversion complete: {success_count}/{len(robots)} succeeded")

    if success_count == len(robots):
        print("\n✓ All conversions successful!")
        print("\nNext steps:")
        print("1. Verify USD files were created in awd/data/assets/*/")
        print("2. Test with: python awd_isaaclab/scripts/run_isaaclab.py --test")
        return 0
    else:
        print("\n✗ Some conversions failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
