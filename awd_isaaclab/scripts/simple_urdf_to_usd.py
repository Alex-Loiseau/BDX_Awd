#!/usr/bin/env python3
"""Simple URDF to USD converter using Isaac Sim's built-in tools."""

import sys
import os

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Get robot from command line
robot = sys.argv[1] if len(sys.argv) > 1 else "go_bdx"

# Setup paths
if robot == "go_bdx":
    urdf_path = os.path.join(project_root, "awd/data/assets/go_bdx/go_bdx.urdf")
    output_dir = os.path.join(project_root, "awd/data/assets/go_bdx")
elif robot == "mini_bdx":
    urdf_path = os.path.join(project_root, "awd/data/assets/mini_bdx/urdf/bdx.urdf")
    output_dir = os.path.join(project_root, "awd/data/assets/mini_bdx/urdf")
else:
    print(f"Unknown robot: {robot}")
    sys.exit(1)

print(f"Converting {robot} URDF to USD", flush=True)
print(f"URDF: {urdf_path}", flush=True)
print(f"Output: {output_dir}", flush=True)

# Import IsaacSim's URDF importer
from isaacsim.asset.importer.urdf import _urdf as urdf_importer

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Import URDF
print("Starting import...", flush=True)
import_config = urdf_importer.ImportConfig()
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.fix_base = False
import_config.make_default_prim = True
import_config.create_physics_scene = False
import_config.import_inertia_tensor = True
import_config.default_drive_type = urdf_importer.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_position_drive_damping = 1.5
import_config.default_position_drive_stiffness = 40.0
import_config.distance_scale = 1.0

# Set output path
usd_filename = os.path.basename(urdf_path).replace(".urdf", ".usd")
usd_path = os.path.join(output_dir, usd_filename)

# Do the import
result = urdf_importer.acquire_urdf_interface().parse_urdf(urdf_path, import_config)

if result:
    print(f"✅ Conversion successful!", flush=True)
    print(f"   USD file: {usd_path}", flush=True)
else:
    print(f"❌ Conversion failed!", flush=True)
    sys.exit(1)
