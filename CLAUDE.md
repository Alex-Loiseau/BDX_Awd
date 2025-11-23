# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWD (Adversarial Waddle Dynamics) - A mallard version of ASE for robotic locomotion training using AMP (Adversarial Motion Priors) on duck-like robots (Open Duck Mini and go_duck).

The project is migrating from rl-games to RSL-RL (Robot Systems Lab - Reinforcement Learning) for Isaac Lab integration.

## Key Commands

### Environment Setup
```bash
# Activate Isaac Sim environment (REQUIRED before any command)
source /home/alexandre/Developpements/IsaacLab/_isaac_sim/python.sh

# Alternative activation methods
/home/alexandre/Developpements/IsaacLab/_isaac_sim/python.sh <script.py>
./run_with_isaaclab.sh <script.py> [args]
```

### Training Commands

#### AWD Training (New RSL-RL Implementation)
```bash
# Quick test (4 environments)
python awd_isaaclab/scripts/train_awd.py --task DucklingCommand --robot go_bdx --num_envs 4 --max_iterations 100 --headless

# Full training (4096 environments)
python awd_isaaclab/scripts/train_awd.py --task DucklingCommand --robot go_bdx --num_envs 4096 --max_iterations 100000 --headless

# With visualization
python awd_isaaclab/scripts/train_awd.py --task DucklingCommand --robot go_bdx --num_envs 16
```

#### RL-Games Training (Legacy)
```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/run_isaaclab.py --task DucklingCommand --num_envs 4096 --robot go_bdx --train --headless
```

### Testing and Visualization
```bash
# View URDF models
python view_urdf.py awd/data/assets/go_bdx/go_bdx.urdf
python view_urdf.py awd/data/assets/mini_bdx/urdf/bdx.urdf

# View TensorBoard logs
tensorboard --logdir logs/awd/DucklingCommand_go_bdx/
```

## Architecture

### Directory Structure
```
awd_isaaclab/           # New Isaac Lab + RSL-RL implementation
├── configs/            # Configuration dataclasses
│   ├── agents/         # Algorithm configs (PPO, AWD)
│   ├── robots/         # Robot configurations
│   └── train/          # Training configurations
├── envs/               # Environment implementations
├── learning/           # AWD algorithm components
├── scripts/            # Training/evaluation scripts
└── utils/              # Utilities

old_awd/                # Legacy rl-games implementation
├── data/               # Assets, configs, motions
├── env/                # Old environment code
└── learning/           # Old learning algorithms

awd/                    # Shared assets
└── data/
    ├── assets/         # URDF files and meshes
    ├── cfg/            # YAML configurations
    └── motions/        # Motion capture JSON files
```

### Key Components

#### Algorithms (awd_isaaclab/learning/)
- **awd_ppo.py**: AWD PPO algorithm with discriminator and encoder
- **awd_models.py**: Neural network architectures
- **awd_storage.py**: Custom rollout storage for AWD
- **awd_runner.py**: Training loop runner
- **amp_replay_buffer.py**: Replay buffers for AMP

#### Environments (awd_isaaclab/envs/)
- **duckling_base_env.py**: Base environment
- **duckling_command_env.py**: Velocity command tracking
- **duckling_heading_env.py**: Heading control
- **duckling_command_amp_env.py**: AMP-enabled command env
- **amp_observations.py**: AMP observation utilities

#### Configurations
- AWD hyperparameters preserved from old code:
  - disc_coef: 5.0
  - enc_coef: 5.0
  - latent_dim: 64
  - task_reward_w: 0.0
  - disc_reward_w: 0.5
  - enc_reward_w: 0.5

## Migration Status

### Current State
- ✅ AWD algorithm fully ported to RSL-RL
- ✅ All network architectures implemented
- ✅ AMP observations computation
- ✅ Training scripts ready
- ⚠️ Motion library loader pending (fetch_amp_obs_demo returns zeros)
- ⚠️ rl-games implementation has observation space issues

### Known Issues and Solutions

#### RL-Games Observation Space Error
**Problem**: `The obs returned by the reset() method was expecting a numpy array`

**Solution**: Do NOT manually define observation_space in environment __init__. Let DirectRLEnv handle it automatically.

#### RSL-RL Training
Training works but motion library is not yet implemented. The discriminator trains on agent observations only.

## Observation Sizes

| Environment Mode | Base Obs | Task Obs | Total | Source |
|-----------------|----------|----------|-------|---------|
| Base/Duckling | 51 | 0 | **51** | duckling.py |
| DucklingCommand | 51 | 3 | **54** | commands: [vx, vy, vyaw] |
| DucklingHeading | 51 | 5 | **56** | [target_dir:2, speed:1, face_dir:2] |
| DucklingAMP/Perturb | 51 | 0 | **51** | No task obs |

Base observations (51 dims):
- projected_gravity: 3
- dof_pos: 16
- dof_vel: 16
- prev_actions: 16

## Development Guidelines

### Important Rules
1. **Always check old_awd/ when uncertain** - The original implementation is the reference
2. **Activate Isaac Sim environment first** - All commands require the Isaac Sim Python environment
3. **Preserve hyperparameters exactly** - Migration should maintain identical training behavior
4. **Use appropriate wrapper scripts** - run_with_isaaclab.sh handles environment setup

### Common Tasks

#### Adding New Environment
1. Inherit from DucklingBaseEnv
2. Override _compute_task_obs() for task-specific observations
3. Update observation size in config
4. Add reward computation

#### Debugging Training
1. Start with small num_envs (4-16) for faster iteration
2. Use --debug flag for verbose output
3. Check TensorBoard for training metrics
4. Logs stored in logs/awd/<task>_<robot>/

### File Mappings (Old → New)
- old_awd/learning/awd_agent.py → awd_isaaclab/learning/awd_ppo.py
- old_awd/learning/awd_network_builder.py → awd_isaaclab/learning/awd_models.py
- old_awd/env/tasks/duckling_amp.py → awd_isaaclab/envs/amp_observations.py
- old_awd/data/cfg/*/train/*.yaml → awd_isaaclab/configs/agents/awd_ppo_cfg.py