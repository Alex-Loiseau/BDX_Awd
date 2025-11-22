#!/usr/bin/env python3
"""Quick test script to verify AMP environments can be instantiated.

This script tests that all AMP-related environments can be created without errors.
It does NOT test motion data loading (which requires actual motion files).

Usage:
    python test_amp_envs.py
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

print("\n" + "="*80)
print("AMP ENVIRONMENTS INSTANTIATION TEST")
print("="*80)
print("This test verifies that AMP environments can be created.")
print("Note: Motion data loading will fail (expected) without actual motion files.")
print("="*80 + "\n")

def test_imports():
    """Test that all AMP modules can be imported."""
    print("1. Testing imports...")

    try:
        from awd_isaaclab.envs.duckling_amp import DucklingAMP, DucklingAMPCfg
        print("   ✅ DucklingAMP imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import DucklingAMP: {e}")
        return False

    try:
        from awd_isaaclab.envs.duckling_amp_task import DucklingAMPTask, DucklingAMPTaskCfg
        print("   ✅ DucklingAMPTask imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import DucklingAMPTask: {e}")
        return False

    try:
        from awd_isaaclab.envs.duckling_view_motion import DucklingViewMotion, DucklingViewMotionCfg
        print("   ✅ DucklingViewMotion imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import DucklingViewMotion: {e}")
        return False

    print()
    return True


def test_configs():
    """Test that all AMP configurations can be created."""
    print("2. Testing configurations...")

    try:
        from awd_isaaclab.envs.duckling_amp import DucklingAMPCfg
        cfg = DucklingAMPCfg()
        print(f"   ✅ DucklingAMPCfg created")
        print(f"      - num_envs: {cfg.scene.num_envs}")
        print(f"      - episode_length_s: {cfg.episode_length_s}")
        print(f"      - state_init: {cfg.state_init}")
        print(f"      - hybrid_init_prob: {cfg.hybrid_init_prob}")
    except Exception as e:
        print(f"   ❌ Failed to create DucklingAMPCfg: {e}")
        return False

    try:
        from awd_isaaclab.envs.duckling_amp_task import DucklingAMPTaskCfg
        cfg = DucklingAMPTaskCfg()
        print(f"   ✅ DucklingAMPTaskCfg created")
        print(f"      - enable_task_obs: {cfg.enable_task_obs}")
    except Exception as e:
        print(f"   ❌ Failed to create DucklingAMPTaskCfg: {e}")
        return False

    try:
        from awd_isaaclab.envs.duckling_view_motion import DucklingViewMotionCfg
        cfg = DucklingViewMotionCfg()
        print(f"   ✅ DucklingViewMotionCfg created")
        print(f"      - pd_control: {cfg.pd_control}")
    except Exception as e:
        print(f"   ❌ Failed to create DucklingViewMotionCfg: {e}")
        return False

    print()
    return True


def test_utilities():
    """Test that AMP utility modules can be imported."""
    print("3. Testing utility modules...")

    try:
        from awd_isaaclab.utils.torch_utils import (
            quat_mul, quat_conjugate, quat_rotate,
            calc_heading, calc_heading_rot, slerp
        )
        print("   ✅ torch_utils imported successfully")
        print("      - quat_mul, quat_conjugate, quat_rotate")
        print("      - calc_heading, calc_heading_rot, slerp")
    except Exception as e:
        print(f"   ❌ Failed to import torch_utils: {e}")
        return False

    try:
        from awd_isaaclab.utils.motion_lib import MotionLib
        print("   ✅ MotionLib imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import MotionLib: {e}")
        return False

    try:
        from awd_isaaclab.utils.bdx.amp_motion_loader import AMPLoader
        print("   ✅ AMPLoader imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import AMPLoader: {e}")
        return False

    try:
        from awd_isaaclab.utils.bdx.pose3d import QuaternionNormalize
        print("   ✅ pose3d imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import pose3d: {e}")
        return False

    try:
        from awd_isaaclab.utils.bdx.motion_util import calc_heading
        print("   ✅ motion_util imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import motion_util: {e}")
        return False

    try:
        from awd_isaaclab.utils.bdx.utils import quaternion_slerp
        print("   ✅ bdx/utils imported successfully")
    except Exception as e:
        print(f"   ❌ Failed to import bdx/utils: {e}")
        return False

    print()
    return True


def test_registration():
    """Test that environments can be registered with run_isaaclab.py."""
    print("4. Testing environment registration in run_isaaclab.py...")

    try:
        with open("awd_isaaclab/scripts/run_isaaclab.py", "r") as f:
            content = f.read()

        # Check for DucklingAMP registration
        if "DucklingAMP" in content and "duckling_amp:DucklingAMP" in content:
            print("   ✅ DucklingAMP registered in run_isaaclab.py")
        else:
            print("   ❌ DucklingAMP NOT found in run_isaaclab.py")
            return False

        # Check for DucklingAMPTask registration
        if "DucklingAMPTask" in content and "duckling_amp_task:DucklingAMPTask" in content:
            print("   ✅ DucklingAMPTask registered in run_isaaclab.py")
        else:
            print("   ❌ DucklingAMPTask NOT found in run_isaaclab.py")
            return False

        # Check for DucklingViewMotion registration
        if "DucklingViewMotion" in content and "duckling_view_motion:DucklingViewMotion" in content:
            print("   ✅ DucklingViewMotion registered in run_isaaclab.py")
        else:
            print("   ❌ DucklingViewMotion NOT found in run_isaaclab.py")
            return False

    except Exception as e:
        print(f"   ❌ Failed to check run_isaaclab.py: {e}")
        return False

    print()
    return True


def test_jit_compilation():
    """Test that JIT-compiled functions can be imported and executed."""
    print("5. Testing JIT-compiled functions...")

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
        key_body_pos = torch.zeros(batch_size, 4, 3)  # 4 key bodies

        # Test build_amp_observations
        amp_obs = build_amp_observations(
            root_pos, root_rot, root_vel, root_ang_vel,
            dof_pos, dof_vel, key_body_pos,
            local_root_obs=True,
            root_height_obs=True
        )

        print(f"   ✅ build_amp_observations JIT function works")
        print(f"      - Input batch size: {batch_size}")
        print(f"      - Output shape: {amp_obs.shape}")
        print(f"      - Expected: ({batch_size}, 138)")

        if amp_obs.shape[1] != 138:
            print(f"   ⚠️  WARNING: Output dimension is {amp_obs.shape[1]}, expected 138")

    except Exception as e:
        print(f"   ❌ Failed JIT test: {e}")
        return False

    print()
    return True


def main():
    """Run all tests."""
    all_passed = True

    # Run tests
    all_passed &= test_imports()
    all_passed &= test_configs()
    all_passed &= test_utilities()
    all_passed &= test_registration()
    all_passed &= test_jit_compilation()

    # Summary
    print("="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nAMP environments are ready for testing with motion data.")
        print("\nNext steps:")
        print("1. Prepare motion data files (JSON format for AMPLoader)")
        print("2. Configure motion_file path in DucklingAMPCfg")
        print("3. Test with: ./run_with_isaaclab.sh DucklingAMP --test --headless")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the errors above before proceeding.")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
