#!/bin/bash
# Test script to verify AMP environments can be imported in IsaacLab environment
# This script uses the IsaacLab Python environment

# Activate IsaacLab environment
source /home/alexandre/Developpements/env_isaaclab/bin/activate

# Run Python test
python3 << 'EOF'
import sys
import os

# Add project to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print("\n" + "="*80)
print("AMP ENVIRONMENTS IMPORT TEST (IsaacLab Environment)")
print("="*80 + "\n")

success = True

# Test 1: Import configurations
print("1. Testing configuration imports...")
try:
    from awd_isaaclab.envs.duckling_amp import DucklingAMPCfg
    print("   ✅ DucklingAMPCfg")
except Exception as e:
    print(f"   ❌ DucklingAMPCfg: {e}")
    success = False

try:
    from awd_isaaclab.envs.duckling_amp_task import DucklingAMPTaskCfg
    print("   ✅ DucklingAMPTaskCfg")
except Exception as e:
    print(f"   ❌ DucklingAMPTaskCfg: {e}")
    success = False

try:
    from awd_isaaclab.envs.duckling_view_motion import DucklingViewMotionCfg
    print("   ✅ DucklingViewMotionCfg")
except Exception as e:
    print(f"   ❌ DucklingViewMotionCfg: {e}")
    success = False

# Test 2: Create configurations
print("\n2. Testing configuration creation...")
try:
    from awd_isaaclab.envs.duckling_amp import DucklingAMPCfg
    cfg = DucklingAMPCfg()
    print(f"   ✅ DucklingAMPCfg created (num_envs={cfg.scene.num_envs})")
except Exception as e:
    print(f"   ❌ Failed to create DucklingAMPCfg: {e}")
    success = False

try:
    from awd_isaaclab.envs.duckling_amp_task import DucklingAMPTaskCfg
    cfg = DucklingAMPTaskCfg()
    print(f"   ✅ DucklingAMPTaskCfg created")
except Exception as e:
    print(f"   ❌ Failed to create DucklingAMPTaskCfg: {e}")
    success = False

try:
    from awd_isaaclab.envs.duckling_view_motion import DucklingViewMotionCfg
    cfg = DucklingViewMotionCfg()
    print(f"   ✅ DucklingViewMotionCfg created (pd_control={cfg.pd_control})")
except Exception as e:
    print(f"   ❌ Failed to create DucklingViewMotionCfg: {e}")
    success = False

# Test 3: Import utilities
print("\n3. Testing utility imports...")
try:
    from awd_isaaclab.utils.torch_utils import quat_mul, calc_heading, slerp
    print("   ✅ torch_utils (quat_mul, calc_heading, slerp)")
except Exception as e:
    print(f"   ❌ torch_utils: {e}")
    success = False

try:
    from awd_isaaclab.utils.motion_lib import MotionLib
    print("   ✅ MotionLib")
except Exception as e:
    print(f"   ❌ MotionLib: {e}")
    success = False

try:
    from awd_isaaclab.utils.bdx.amp_motion_loader import AMPLoader
    print("   ✅ AMPLoader")
except Exception as e:
    print(f"   ❌ AMPLoader: {e}")
    success = False

# Test 4: Test JIT functions
print("\n4. Testing JIT-compiled functions...")
try:
    import torch
    from awd_isaaclab.envs.duckling_amp import build_amp_observations

    # Create dummy data
    batch_size = 4
    root_pos = torch.zeros(batch_size, 3)
    root_rot = torch.zeros(batch_size, 4)
    root_rot[:, 0] = 1.0  # w=1 for identity quaternion
    root_vel = torch.zeros(batch_size, 3)
    root_ang_vel = torch.zeros(batch_size, 3)
    dof_pos = torch.zeros(batch_size, 12)
    dof_vel = torch.zeros(batch_size, 12)
    key_body_pos = torch.zeros(batch_size, 4, 3)

    # Test function
    amp_obs = build_amp_observations(
        root_pos, root_rot, root_vel, root_ang_vel,
        dof_pos, dof_vel, key_body_pos,
        local_root_obs=True, root_height_obs=True
    )

    print(f"   ✅ build_amp_observations works (output shape: {amp_obs.shape})")
except Exception as e:
    print(f"   ❌ JIT function test failed: {e}")
    success = False

# Summary
print("\n" + "="*80)
if success:
    print("✅ ALL IMPORT TESTS PASSED")
    print("\nAMP environments are ready. Next steps:")
    print("1. Prepare motion data files")
    print("2. Test with: ./run_with_isaaclab.sh DucklingAMP --test --headless")
else:
    print("❌ SOME TESTS FAILED")
    print("Please fix errors above.")
print("="*80 + "\n")

sys.exit(0 if success else 1)
EOF
