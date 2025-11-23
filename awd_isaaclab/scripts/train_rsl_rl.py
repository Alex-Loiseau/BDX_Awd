#!/usr/bin/env python3
"""Train Duckling locomotion tasks with RSL-RL.

This script replaces run_isaaclab.py and uses RSL-RL instead of rl-games.

Usage:
    # Training
    python train_rsl_rl.py --task DucklingCommand --robot go_bdx --num_envs 4096

    # Training with specific config
    python train_rsl_rl.py --task DucklingHeading --robot mini_bdx --agent rsl_rl_ppo_cfg

    # Headless training
    python train_rsl_rl.py --task DucklingCommand --robot go_bdx --headless --num_envs 4096
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
parser = argparse.ArgumentParser(description="Train Duckling locomotion with RSL-RL")

# Environment arguments
parser.add_argument("--task", type=str, required=True, help="Task name (e.g., DucklingCommand)")
parser.add_argument("--robot", type=str, default="mini_bdx", choices=["mini_bdx", "go_bdx"],
                    help="Robot type")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")

# Training arguments
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
parser.add_argument("--agent", type=str, default="rsl_rl_ppo_cfg",
                    help="Agent configuration name (without .py extension)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Checkpoint arguments
parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
parser.add_argument("--load_run", type=int, default=-1, help="Run number to load (-1 for latest)")
parser.add_argument("--checkpoint", type=str, default="model", help="Checkpoint file name")

# Video recording arguments
parser.add_argument("--video", action="store_true", help="Record training videos")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between videos")

# Debugging
parser.add_argument("--debug", action="store_true", help="Enable debug mode")

# AppLauncher arguments (includes --headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Always enable cameras if recording video
if args.video:
    args.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after Isaac Sim is launched
import torch
import gymnasium

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg


def create_env_config(args):
    """Create environment configuration based on arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Environment configuration object.
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

    # Import task config
    if args.task == "DucklingCommand":
        from awd_isaaclab.envs.duckling_command_env import DucklingCommandCfg
        cfg = DucklingCommandCfg()

    elif args.task == "DucklingHeading":
        from awd_isaaclab.envs.duckling_heading_env import DucklingHeadingCfg
        cfg = DucklingHeadingCfg()

    elif args.task == "DucklingPerturb":
        from awd_isaaclab.envs.duckling_perturb_env import DucklingPerturbCfg
        cfg = DucklingPerturbCfg()

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Apply robot configuration
    cfg.robot = robot_cfg

    # Apply robot-specific parameters
    cfg.num_actions = robot_params["num_actions"]
    cfg.termination_height = robot_params["termination_height"]
    cfg.head_termination_height = robot_params["head_termination_height"]
    cfg.scene.env_spacing = robot_params["env_spacing"]

    # Apply command ranges (for Command, Heading, and Perturb tasks only)
    if hasattr(cfg, 'command_x_range'):
        cfg.command_x_range = tuple(robot_params["command_ranges"]["linear_x"])
        cfg.command_y_range = tuple(robot_params["command_ranges"]["linear_y"])
        cfg.command_yaw_range = tuple(robot_params["command_ranges"]["yaw"])

    # Apply reward scales (if task has reward structure)
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

    # Override with command line arguments
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs

    if args.headless:
        cfg.viewer.enabled = False

    # Set seed
    cfg.seed = args.seed

    return cfg


def load_agent_config(agent_name: str, task_name: str, robot_name: str) -> RslRlOnPolicyRunnerCfg:
    """Load agent configuration.

    Args:
        agent_name: Name of agent config module (without .py)
        task_name: Task name (e.g., DucklingCommand)
        robot_name: Robot name (e.g., go_bdx)

    Returns:
        Agent configuration object.
    """
    # Import agent config
    try:
        agent_module = __import__(f"awd_isaaclab.configs.agents.{agent_name}", fromlist=["*"])
    except ImportError as e:
        print(f"ERROR: Could not import agent config '{agent_name}': {e}")
        print(f"Available configs should be in: awd_isaaclab/configs/agents/")
        sys.exit(1)

    # Get config class based on task name
    if task_name == "DucklingCommand":
        cfg_class_name = "DucklingCommandPPORunnerCfg"
    elif task_name == "DucklingHeading":
        cfg_class_name = "DucklingHeadingPPORunnerCfg"
    elif task_name == "DucklingPerturb":
        cfg_class_name = "DucklingPerturbPPORunnerCfg"
    else:
        # Fallback to generic config
        cfg_class_name = "DucklingCommandPPORunnerCfg"

    # Get config class
    if not hasattr(agent_module, cfg_class_name):
        print(f"WARNING: Config class '{cfg_class_name}' not found in {agent_name}")
        print(f"Falling back to DucklingCommandPPORunnerCfg")
        cfg_class_name = "DucklingCommandPPORunnerCfg"

    agent_cfg = getattr(agent_module, cfg_class_name)()

    # Update experiment name to include robot
    agent_cfg.experiment_name = f"{task_name}_{robot_name}"

    # Apply command line overrides
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    if args.resume:
        agent_cfg.resume = True
        agent_cfg.load_run = args.load_run
        agent_cfg.load_checkpoint = args.checkpoint

    agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    return agent_cfg


def main():
    """Main training function."""
    print("="*80)
    print(f"Training {args.task} with {args.robot} using RSL-RL")
    print("="*80)

    # Create environment configuration
    env_cfg = create_env_config(args)

    # Load agent configuration
    agent_cfg = load_agent_config(args.agent, args.task, args.robot)

    # Specify directory for logging
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # Create log directory with timestamp
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Register environment
    task_name = f"{args.task}_{args.robot}"

    # Determine entry point
    if args.task == "DucklingCommand":
        entry_point = "awd_isaaclab.envs.duckling_command_env:DucklingCommandEnv"
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
    render_mode = "rgb_array" if args.video else None
    env = gymnasium.make(task_name, cfg=env_cfg, render_mode=render_mode)
    print(f"[INFO] Environment created: {env_cfg.scene.num_envs} parallel environments")

    # Wrap for video recording if requested
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training")
        print_dict(video_kwargs, nesting=4)
        env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    print("[INFO] Wrapping environment for RSL-RL...")
    env = RslRlVecEnvWrapper(env)

    # Print configuration summary
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Robot: {args.robot}")
    print(f"Num Envs: {env_cfg.scene.num_envs}")
    print(f"Max Iterations: {agent_cfg.max_iterations}")
    print(f"Device: {agent_cfg.device}")
    print(f"Log Directory: {log_dir}")
    print("="*80 + "\n")

    # Create RSL-RL runner
    print("[INFO] Creating RSL-RL runner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Start training
    print("[INFO] Starting training...")
    print("="*80)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

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
