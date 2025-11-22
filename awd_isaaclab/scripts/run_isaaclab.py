#!/usr/bin/env python3
"""Main script to run Duckling tasks with IsaacLab.

This is the IsaacLab version of awd/run.py.
Supports training and playing with rl-games integration.

Usage:
    # Training
    python run_isaaclab.py --task DucklingCommand --robot mini_bdx --train

    # Playing (inference)
    python run_isaaclab.py --task DucklingCommand --robot mini_bdx --play --checkpoint path/to/checkpoint.pth

    # Headless training
    python run_isaaclab.py --task DucklingCommand --robot mini_bdx --train --headless --num_envs 4096
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import torch
import gymnasium as gym

# Import IsaacLab (version 0.48.4+ uses 'isaaclab' namespace directly)
# Note: Older versions used 'omni.isaac.lab', newer versions use 'isaaclab'
try:
    from isaaclab.app import AppLauncher
except ImportError:
    # Fallback for older versions
    from omni.isaac.lab.app import AppLauncher


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Duckling locomotion tasks with IsaacLab")

    # Environment arguments
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., DucklingCommand)")
    parser.add_argument("--robot", type=str, default="mini_bdx", choices=["mini_bdx", "go_bdx"],
                        help="Robot type")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")

    # Mode arguments
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--play", action="store_true", help="Run inference/play mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for playing")

    # Training arguments
    parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum training iterations")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")

    # Rendering arguments
    parser.add_argument("--headless", action="store_true", help="Run headless (no visualization)")
    parser.add_argument("--video", action="store_true", help="Record video during play")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")

    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--test", action="store_true", help="Run a quick test (100 steps)")

    return parser.parse_args()


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

    elif args.task == "DucklingAMP":
        from awd_isaaclab.envs.duckling_amp import DucklingAMPCfg
        cfg = DucklingAMPCfg()

    elif args.task == "DucklingAMPTask":
        from awd_isaaclab.envs.duckling_amp_task import DucklingAMPTaskCfg
        cfg = DucklingAMPTaskCfg()

    elif args.task == "DucklingViewMotion":
        from awd_isaaclab.envs.duckling_view_motion import DucklingViewMotionCfg
        cfg = DucklingViewMotionCfg()

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Apply robot configuration
    cfg.robot = robot_cfg

    # Apply robot-specific parameters
    # Note: AMP tasks have different num_observations, so only apply if not AMP
    if not args.task.startswith("DucklingAMP") and args.task != "DucklingViewMotion":
        cfg.num_observations = robot_params["num_observations"]

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

    # Test mode: reduce envs and episode length
    if args.test:
        cfg.scene.num_envs = 16
        cfg.episode_length_s = 5.0

    return cfg


def create_rlgames_config(args, env):
    """Create rl-games configuration.

    Args:
        args: Command line arguments.
        env: Environment instance.

    Returns:
        Dictionary with rl-games configuration.
    """
    # Get underlying environment
    unwrapped_env = env.unwrapped

    # Create experiment name
    if args.experiment is None:
        experiment_name = f"{args.task}_{args.robot}"
    else:
        experiment_name = args.experiment

    # Create run name
    if args.run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_name = args.run_name

    # Base configuration
    config = {
        "params": {
            "seed": 42,
            "algo": {
                "name": "a2c_continuous",
            },
            "model": {
                "name": "continuous_a2c_logstd",
            },
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {
                            "name": "default",
                        },
                        "sigma_init": {
                            "name": "const_initializer",
                            "val": 0,
                        },
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [512, 256, 128],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {
                        "name": "default",
                    },
                    "regularizer": {
                        "name": "None",
                    },
                },
            },
            "load_checkpoint": args.checkpoint is not None,
            "load_path": args.checkpoint if args.checkpoint else "",
            "config": {
                "name": experiment_name,
                "full_experiment_name": f"{experiment_name}_{run_name}",
                "env_name": "rlgpu",
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": unwrapped_env.num_envs,
                "reward_shaper": {
                    "scale_value": 1.0,
                },
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                "lr_schedule": "adaptive",
                "kl_threshold": 0.008,
                "score_to_win": 20000,
                "max_epochs": args.max_iterations,
                "save_best_after": 100,
                "save_frequency": 50,
                "print_stats": True,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": 32,
                "minibatch_size": 32768,
                "mini_epochs": 5,
                "critic_coef": 4,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001,
            },
        }
    }

    return config


# Global variable to store the pre-created environment for rl-games
_global_env = None


def train(env, args):
    """Run training with rl-games.

    Args:
        env: Environment instance (Gymnasium).
        args: Command line arguments.
    """
    global _global_env

    try:
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
    except ImportError:
        print("ERROR: rl-games not installed!")
        print("Install with: pip install rl-games")
        sys.exit(1)

    # Define RLGPUEnv wrapper inside train() so vecenv is in scope
    class RLGPUEnv(vecenv.IVecEnv):
        """Wrapper to adapt IsaacLab Gymnasium environment to rl-games API.

        This is a faithful port of the RLGPUEnv class from awd/run.py.
        It converts the Gymnasium API (5 return values) to the Gym API (4 return values)
        expected by rl-games.

        Note: In the original IsaacGym code, this wrapper created the environment.
        With IsaacLab, we must create the environment before rl-games (for AppLauncher),
        so we use a global variable to pass the pre-created environment.
        """
        def __init__(self, config_name, num_actors, **kwargs):
            global _global_env

            # Use the pre-created environment
            self.env = _global_env

            # Check if environment has state space (for critic with global state)
            self.use_global_obs = hasattr(self.env.unwrapped, 'num_states') and self.env.unwrapped.num_states > 0

            # Initialize state dict
            # Call reset() to get initial observation (Gymnasium envs need explicit reset())
            self.full_state = {}
            self.full_state["obs"] = self.reset()
            if self.use_global_obs:
                self.full_state["states"] = self.env.unwrapped.get_state()
            return

        def step(self, action):
            # Gymnasium API: obs, reward, terminated, truncated, info
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Convert to Gym API: done = terminated | truncated
            is_done = terminated | truncated

            # Extract "policy" obs if obs is a dict (IsaacLab returns dict with "policy" key)
            if isinstance(next_obs, dict):
                next_obs = next_obs["policy"]

            # Update state
            self.full_state["obs"] = next_obs
            if self.use_global_obs:
                self.full_state["states"] = self.env.unwrapped.get_state()
                return self.full_state, reward, is_done, info
            else:
                return self.full_state["obs"], reward, is_done, info

        def reset(self, env_ids=None):
            # Always call the real reset - Gymnasium envs need explicit reset() after gym.make()
            print(f"[DEBUG WRAPPER] reset() called, env_ids={env_ids}")
            print("[DEBUG WRAPPER] Calling env.reset()")
            obs, info = self.env.reset()
            print(f"[DEBUG WRAPPER] env.reset() returned obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

            # Extract "policy" obs if obs is a dict (IsaacLab returns dict with "policy" key)
            if isinstance(obs, dict):
                obs = obs["policy"]
                print(f"[DEBUG WRAPPER] Extracted 'policy' obs, shape: {obs.shape}")

            self.full_state["obs"] = obs
            if self.use_global_obs:
                self.full_state["states"] = self.env.unwrapped.get_state()
                print("[DEBUG WRAPPER] Returning full_state with states")
                return self.full_state
            else:
                print("[DEBUG WRAPPER] Returning obs only")
                return self.full_state["obs"]

        def get_number_of_agents(self):
            return self.env.unwrapped.num_envs

        def get_env_info(self):
            info = {}
            unwrapped = self.env.unwrapped

            # Action space
            info["action_space"] = unwrapped.single_action_space

            # Observation space - extract "policy" if it's a Dict space
            obs_space = unwrapped.single_observation_space
            if hasattr(obs_space, 'spaces') and 'policy' in obs_space.spaces:
                # This is a Dict space with "policy" key - extract it
                info["observation_space"] = obs_space.spaces['policy']
            else:
                info["observation_space"] = obs_space

            # AMP observation space (if available)
            if hasattr(unwrapped, 'amp_observation_space'):
                info["amp_observation_space"] = unwrapped.amp_observation_space

            # State space (if available)
            if self.use_global_obs:
                info["state_space"] = unwrapped.state_space
                print(info["action_space"], info["observation_space"], info["state_space"])
            else:
                print(info["action_space"], info["observation_space"])

            return info

    def create_rlgpu_env(**kwargs):
        """Create environment for rl-games.

        This returns the pre-created Gymnasium environment stored in _global_env.
        In the original IsaacGym code, this would create a new environment,
        but with IsaacLab we must create it before rl-games starts.
        """
        global _global_env
        return _global_env

    # Store environment in global variable for the wrapper
    _global_env = env

    # Register vecenv wrapper (like in awd/run.py)
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
    )

    # Register environment configuration (like in awd/run.py)
    env_configurations.register(
        "rlgpu",
        {
            "env_creator": lambda **kwargs: create_rlgpu_env(**kwargs),
            "vecenv_type": "RLGPU",
        },
    )

    # Create rl-games config
    rl_config = create_rlgames_config(args, env)

    # Create runner
    runner = Runner()
    runner.load(rl_config)
    runner.reset()

    # Run training
    runner.run({
        "train": True,
        "play": False,
        "checkpoint": args.checkpoint,
        "sigma": None,
    })


def test_environment(env, args):
    """Test environment with random actions.

    Args:
        env: Environment instance.
        args: Command line arguments.
    """
    # Get underlying environment
    unwrapped_env = env.unwrapped

    print("\n" + "="*80)
    print("TEST MODE - Running environment with random actions")
    print("="*80)
    print(f"Num envs: {unwrapped_env.num_envs}")
    print(f"Num observations: {unwrapped_env.num_observations}")
    print(f"Num actions: {unwrapped_env.num_actions}")
    print("="*80 + "\n")

    # Reset environment
    obs, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")

    # Run a few steps with random actions
    num_steps = 100  # Short test
    print(f"\n🚀 Running {num_steps} steps with random actions...")

    for step in range(num_steps):
        # Random actions
        actions = torch.randn((unwrapped_env.num_envs, unwrapped_env.num_actions), device=args.device) * 0.1

        # Step environment (Gymnasium API: returns 5 values)
        obs, reward, terminated, truncated, info = env.step(actions)

        if step % 20 == 0:
            print(f"   Step {step}: Mean reward = {reward.mean().item():.4f}")

    print("\n✅ Test completed successfully!")
    print(f"   Final mean reward: {reward.mean().item():.4f}")
    print(f"   Environment is working correctly!")


def play(env, args):
    """Run inference/play mode.

    Args:
        env: Environment instance.
        args: Command line arguments.
    """
    # Si mode test sans checkpoint, juste tester l'environnement avec actions aléatoires
    if args.test and args.checkpoint is None:
        print("INFO: Test mode - running with random actions")
        test_environment(env, args)
        return

    if args.checkpoint is None:
        print("ERROR: Must provide --checkpoint for play mode!")
        sys.exit(1)

    # Get underlying environment
    unwrapped_env = env.unwrapped

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Create policy network
    # TODO: Implement policy loading

    print(f"Playing with checkpoint: {args.checkpoint}")

    # Run episodes
    obs, info = env.reset()
    episode_rewards = []
    episode_reward = 0

    num_episodes = 10 if not args.test else 1

    for episode in range(num_episodes):
        done = False
        obs, info = env.reset()

        while not done:
            # TODO: Get action from policy
            action = torch.zeros((unwrapped_env.num_envs, unwrapped_env.num_actions), device=args.device)

            # Gymnasium API: returns 5 values
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()  # Episode ends if any env terminates/truncates
            episode_reward += reward.mean().item()

            if args.video:
                # TODO: Record video
                pass

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        episode_reward = 0

    print(f"\nAverage reward over {num_episodes} episodes: {sum(episode_rewards) / len(episode_rewards):.2f}")


def main():
    """Main entry point."""
    print("[DEBUG] main() called - starting program...")
    # Parse arguments
    args = parse_args()
    print(f"[DEBUG] Arguments parsed: task={args.task}, train={args.train}, play={args.play}")

    # Check that either train or play is specified
    # If --test is specified without --train or --play, default to --play mode
    if args.test and not args.train and not args.play:
        args.play = True
        print("INFO: --test mode enabled, using --play mode")

    if not args.train and not args.play:
        print("ERROR: Must specify either --train or --play (or use --test)")
        sys.exit(1)

    if args.train and args.play:
        print("ERROR: Cannot specify both --train and --play")
        sys.exit(1)

    # Launch Isaac Sim app
    launcher = AppLauncher(args_cli=[
        "--headless" if args.headless else "",
        f"--/renderer/enabled={'False' if args.headless else 'True'}",
    ])
    simulation_app = launcher.app

    # Create environment configuration
    env_cfg = create_env_config(args)

    # Register environment with correct entry point
    task_name = f"{args.task}_{args.robot}"

    # Determine entry point based on task
    if args.task == "DucklingCommand":
        entry_point = "awd_isaaclab.envs.duckling_command_env:DucklingCommandEnv"
    elif args.task == "DucklingHeading":
        entry_point = "awd_isaaclab.envs.duckling_heading_env:DucklingHeadingEnv"
    elif args.task == "DucklingPerturb":
        entry_point = "awd_isaaclab.envs.duckling_perturb_env:DucklingPerturbEnv"
    elif args.task == "DucklingAMP":
        entry_point = "awd_isaaclab.envs.duckling_amp:DucklingAMP"
    elif args.task == "DucklingAMPTask":
        entry_point = "awd_isaaclab.envs.duckling_amp_task:DucklingAMPTask"
    elif args.task == "DucklingViewMotion":
        entry_point = "awd_isaaclab.envs.duckling_view_motion:DucklingViewMotion"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    print(f"[DEBUG] Registering environment {task_name} with entry_point {entry_point}")
    gym.register(
        id=task_name,
        entry_point=entry_point,
        kwargs={"cfg": env_cfg},
    )
    print(f"[DEBUG] Environment registered successfully")

    # Create environment
    print("[DEBUG] Creating environment...")
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args.headless else "human")
    print("[DEBUG] Environment created successfully!")

    # Do an initial reset to fully initialize the simulation
    # This is needed before creating the rl-games wrapper
    print("[DEBUG] Performing initial reset to initialize simulation...")
    initial_obs, initial_info = env.reset()
    if isinstance(initial_obs, dict):
        print(f"[DEBUG] Initial reset done, obs keys: {initial_obs.keys()}")
    else:
        print(f"[DEBUG] Initial reset done, obs shape: {initial_obs.shape}")

    print(f"\n{'='*80}")
    print(f"Task: {args.task}")
    print(f"Robot: {args.robot}")
    print(f"Num Envs: {env_cfg.scene.num_envs}")
    print(f"Num Observations: {env_cfg.num_observations}")
    print(f"Num Actions: {env_cfg.num_actions}")
    print(f"Device: {args.device}")
    print(f"Mode: {'Training' if args.train else 'Playing'}")
    print(f"{'='*80}\n")

    try:
        print("[DEBUG] Starting training/playing...")
        if args.train:
            train(env, args)
        else:
            play(env, args)
    finally:
        # Cleanup
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
