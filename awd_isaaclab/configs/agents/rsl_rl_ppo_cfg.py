# Copyright (c) 2024, BDX AWD Project
# All rights reserved.

"""RSL-RL PPO configuration for AWD Duckling locomotion tasks.

This configuration matches the hyperparameters from old_awd/data/cfg/go_bdx/train/awd_duckling.yaml
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DucklingCommandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for DucklingCommand PPO training.

    Hyperparameters match the original rl-games config:
    - old_awd/data/cfg/go_bdx/train/awd_duckling.yaml

    Mapping from rl-games to RSL-RL:
    - horizon_length (32) -> num_steps_per_env (32)
    - mini_epochs (6) -> num_learning_epochs (6)
    - minibatch_size (16384) -> Calculated from num_mini_batches
    - learning_rate (2e-5) -> learning_rate (2e-5)
    - gamma (0.99) -> gamma (0.99)
    - tau (0.95) -> lam (0.95)
    - e_clip (0.2) -> clip_param (0.2)
    - entropy_coef (0.0) -> entropy_coef (0.0)
    - critic_coef (5) -> value_loss_coef (5.0)
    - grad_norm (1.0) -> max_grad_norm (1.0)
    - max_epochs (100000) -> max_iterations (100000)
    """

    # Experiment settings
    num_steps_per_env = 32  # horizon_length in rl-games
    max_iterations = 100000  # max_epochs in rl-games
    save_interval = 1000  # save_frequency in rl-games
    experiment_name = "duckling_command"

    # Resume and checkpoint settings
    resume = False
    load_run = -1
    load_checkpoint = "model"  # Checkpoint file name

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None

    # Policy network configuration
    policy = RslRlPpoActorCriticCfg(
        # Network architecture - matching old config [1024, 1024, 512]
        init_noise_std=0.054881163609743834,  # exp(-2.9) from sigma_init in rl-games config
        actor_obs_normalization=True,  # normalize_input: True
        critic_obs_normalization=True,  # normalize_value: True
        actor_hidden_dims=[1024, 1024, 512],  # mlp.units in rl-games
        critic_hidden_dims=[1024, 1024, 512],  # Same as actor for symmetric network
        activation="relu",  # mlp.activation in rl-games
        # Actor/critic are separated (separate: True in rl-games)
    )

    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        # Core PPO settings
        value_loss_coef=5.0,  # critic_coef in rl-games
        use_clipped_value_loss=False,  # clip_value: False in rl-games
        clip_param=0.2,  # e_clip in rl-games
        entropy_coef=0.0,  # entropy_coef in rl-games

        # Training settings
        num_learning_epochs=6,  # mini_epochs in rl-games
        # num_mini_batches calculated from minibatch_size
        # With num_envs=4096, num_steps_per_env=32 -> total_samples = 131072
        # minibatch_size=16384 -> num_mini_batches = 131072/16384 = 8
        num_mini_batches=8,  # Calculated from minibatch_size (16384)

        # Learning rate and schedule
        learning_rate=2e-5,  # learning_rate in rl-games
        schedule="constant",  # lr_schedule: constant in rl-games

        # GAE settings
        gamma=0.99,  # gamma in rl-games
        lam=0.95,  # tau (GAE lambda) in rl-games

        # Gradient settings
        desired_kl=0.008,  # Not in original, using RSL-RL default (for adaptive LR)
        max_grad_norm=1.0,  # grad_norm in rl-games
    )


@configclass
class DucklingHeadingPPORunnerCfg(DucklingCommandPPORunnerCfg):
    """Configuration for DucklingHeading PPO training.

    Inherits all settings from DucklingCommand and only changes experiment name.
    """
    experiment_name = "duckling_heading"


@configclass
class DucklingPerturbPPORunnerCfg(DucklingCommandPPORunnerCfg):
    """Configuration for DucklingPerturb PPO training.

    Inherits all settings from DucklingCommand and only changes experiment name.
    """
    experiment_name = "duckling_perturb"
