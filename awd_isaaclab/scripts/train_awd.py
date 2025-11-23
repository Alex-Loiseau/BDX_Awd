#!/usr/bin/env python3
"""Train Duckling locomotion tasks with AWD (AMP with Diversity).

This script uses the custom AWD runner and algorithm for style-imitative locomotion.

Usage:
    # Training with AWD
    python train_awd.py --task DucklingCommand --robot go_bdx --num_envs 4096

    # Headless training
    python train_awd.py --task DucklingCommand --robot go_bdx --headless --num_envs 4096

    # Resume training
    python train_awd.py --task DucklingCommand --robot go_bdx --resume --load_run 0
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train Duckling locomotion with AWD")

# Environment arguments
parser.add_argument("--task", type=str, required=True, help="Task name (e.g., DucklingCommand)")
parser.add_argument("--robot", type=str, default="go_bdx", choices=["mini_bdx", "go_bdx"],
                    help="Robot type")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")

# Training arguments
parser.add_argument("--max_iterations", type=int, default=100000, help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Checkpoint arguments
parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
parser.add_argument("--load_run", type=int, default=-1, help="Run number to load (-1 for latest)")
parser.add_argument("--checkpoint", type=str, default="model", help="Checkpoint file name")

# Debugging
parser.add_argument("--debug", action="store_true", help="Enable debug mode")

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after Isaac Sim is launched
import torch
import gymnasium

from awd_isaaclab.learning import AWDOnPolicyRunner
from isaaclab.utils.dict import print_dict


def create_env_config(args):
    """Create environment configuration.

    Args:
        args: Parsed arguments.

    Returns:
        Environment configuration.
    """
    # Import robot configs
    if args.robot == "mini_bdx":
        from awd_isaaclab.configs.robots.mini_bdx_cfg import MINI_BDX_CFG, MINI_BDX_PARAMS
        robot_cfg = MINI_BDX_CFG
        robot_params = MINI_BDX_PARAMS
    elif args.robot == "go_bdx":
        from awd_isaaclab.configs.robots.go_bdx_cfg import GO_BDX_CFG, GO_BDX_PARAMS
        robot_cfg = GO_BDX_CFG
        robot_params = GO_BDX_PARAMS
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    # Import task config (use AMP versions for AWD training)
    if args.task == "DucklingCommand":
        from awd_isaaclab.envs.duckling_command_amp_env import DucklingCommandAMPCfg
        cfg = DucklingCommandAMPCfg()
    elif args.task == "DucklingHeading":
        # TODO: Create DucklingHeadingAMPEnv
        from awd_isaaclab.envs.duckling_heading_env import DucklingHeadingCfg
        cfg = DucklingHeadingCfg()
    elif args.task == "DucklingPerturb":
        # TODO: Create DucklingPerturbAMPEnv
        from awd_isaaclab.envs.duckling_perturb_env import DucklingPerturbCfg
        cfg = DucklingPerturbCfg()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Apply robot configuration
    cfg.robot = robot_cfg
    cfg.num_actions = robot_params["num_actions"]
    cfg.termination_height = robot_params["termination_height"]
    cfg.head_termination_height = robot_params["head_termination_height"]
    cfg.scene.env_spacing = robot_params["env_spacing"]

    # Apply command ranges
    if hasattr(cfg, 'command_x_range'):
        cfg.command_x_range = tuple(robot_params["command_ranges"]["linear_x"])
        cfg.command_y_range = tuple(robot_params["command_ranges"]["linear_y"])
        cfg.command_yaw_range = tuple(robot_params["command_ranges"]["yaw"])

    # Apply reward scales
    if hasattr(cfg, 'lin_vel_xy_reward_scale'):
        cfg.lin_vel_xy_reward_scale = robot_params["reward_scales"]["lin_vel_xy"]
        cfg.ang_vel_z_reward_scale = robot_params["reward_scales"]["ang_vel_z"]
        cfg.torque_reward_scale = robot_params["reward_scales"]["torque"]
        cfg.action_rate_reward_scale = robot_params["reward_scales"]["action_rate"]
        cfg.stand_still_reward_scale = robot_params["reward_scales"]["stand_still"]

    # Apply normalization scales
    if hasattr(cfg, 'lin_vel_scale'):
        cfg.lin_vel_scale = robot_params["lin_vel_scale"]
    if hasattr(cfg, 'ang_vel_scale'):
        cfg.ang_vel_scale = robot_params["ang_vel_scale"]

    # Override with command line
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs

    if args.headless:
        cfg.viewer.enabled = False

    cfg.seed = args.seed

    # Add AMP-specific settings
    cfg.num_amp_obs_steps = 2

    return cfg


def load_awd_config(task_name: str, robot_name: str) -> dict:
    """Load AWD configuration.

    Args:
        task_name: Task name.
        robot_name: Robot name.

    Returns:
        AWD configuration dictionary.
    """
    from awd_isaaclab.configs.agents.awd_ppo_cfg import AWDPPORunnerCfg

    # Create config
    agent_cfg = AWDPPORunnerCfg()
    agent_cfg.experiment_name = f"{task_name}_{robot_name}_AWD"

    # Apply command line overrides
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    if args.resume:
        agent_cfg.resume = True
        agent_cfg.load_run = args.load_run
        agent_cfg.load_checkpoint = args.checkpoint

    agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    return agent_cfg.to_dict()


def main():
    """Main training function."""
    print("=" * 80)
    print(f"Training {args.task} with {args.robot} using AWD")
    print("=" * 80)

    # Create environment configuration
    env_cfg = create_env_config(args)

    # Load AWD configuration
    awd_cfg = load_awd_config(args.task, args.robot)

    # Adjust num_mini_batches based on num_envs to ensure it divides evenly
    num_envs = env_cfg.scene.num_envs
    num_steps = awd_cfg["num_steps_per_env"]
    total_steps = num_envs * num_steps

    # Find a suitable num_mini_batches that divides total_steps
    original_mini_batches = awd_cfg["algorithm"]["num_mini_batches"]
    if total_steps < original_mini_batches:
        # For small number of environments, use 1 mini-batch
        awd_cfg["algorithm"]["num_mini_batches"] = 1
    else:
        # Find the largest divisor of total_steps that is <= original_mini_batches
        for i in range(original_mini_batches, 0, -1):
            if total_steps % i == 0:
                awd_cfg["algorithm"]["num_mini_batches"] = i
                break

    if awd_cfg["algorithm"]["num_mini_batches"] != original_mini_batches:
        print(f"[INFO] Adjusted num_mini_batches from {original_mini_batches} to {awd_cfg['algorithm']['num_mini_batches']} for {num_envs} environments")

    # Specify directory for logging
    log_root_path = os.path.join("logs", "awd", f"{args.task}_{args.robot}")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # Create log directory with timestamp
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if awd_cfg.get("run_name"):
        log_dir += f"_{awd_cfg['run_name']}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Register environment
    task_name = f"{args.task}_{args.robot}_AWD"

    # Determine entry point (use AMP versions)
    if args.task == "DucklingCommand":
        entry_point = "awd_isaaclab.envs.duckling_command_amp_env:DucklingCommandAMPEnv"
    elif args.task == "DucklingHeading":
        entry_point = "awd_isaaclab.envs.duckling_heading_env:DucklingHeadingEnv"
    elif args.task == "DucklingPerturb":
        entry_point = "awd_isaaclab.envs.duckling_perturb_env:DucklingPerturbEnv"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"[INFO] Registering environment: {task_name}")
    gymnasium.register(
        id=task_name,
        entry_point=entry_point,
        kwargs={"cfg": env_cfg},
    )

    # Create environment
    print(f"[INFO] Creating environment...")
    env = gymnasium.make(task_name, cfg=env_cfg)
    print(f"[INFO] Environment created: {env_cfg.scene.num_envs} parallel environments")

    # Wrap for AWD (custom wrapper needed)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    print("[INFO] Wrapping environment for AWD...")
    env = RslRlVecEnvWrapper(env)

    # Print configuration summary
    print("\n" + "=" * 80)
    print("AWD TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Robot: {args.robot}")
    print(f"Num Envs: {env_cfg.scene.num_envs}")
    print(f"Max Iterations: {awd_cfg['max_iterations']}")
    print(f"Device: {awd_cfg['device']}")
    print(f"Log Directory: {log_dir}")
    print(f"Latent Dim: {awd_cfg['algorithm']['latent_dim']}")
    print(f"Disc Coef: {awd_cfg['algorithm']['disc_coef']}")
    print(f"Enc Coef: {awd_cfg['algorithm']['enc_coef']}")
    print("=" * 80 + "\n")

    # Create AWD runner
    print("[INFO] Creating AWD runner...")
    runner = AWDOnPolicyRunner(env, awd_cfg, log_dir=log_dir, device=awd_cfg["device"])

    # Start training
    print("[INFO] Starting AWD training...")
    print("=" * 80)
    runner.learn(num_learning_iterations=awd_cfg["max_iterations"], init_at_random_ep_len=True)

    # Cleanup
    print("\n[INFO] Training completed!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
