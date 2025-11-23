"""AWD PPO Configuration for RSL-RL.

This configuration matches the hyperparameters from old_awd/data/cfg/go_bdx/train/awd_duckling.yaml
"""

from dataclasses import dataclass, field
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@dataclass
class AWDPPOActorCriticCfg(RslRlPpoActorCriticCfg):
    """AWD Actor-Critic network configuration."""

    # Network architecture (matches old config)
    actor_hidden_dims: list = (1024, 1024, 512)
    critic_hidden_dims: list = (1024, 1024, 512)
    activation: str = "relu"

    # Action distribution
    init_noise_std: float = 0.054881163609743834  # exp(-2.9)

    # Normalization
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True

    # AWD-specific network settings
    latent_dim: int = 64
    style_dim: int = 64
    style_hidden_dims: list = (512, 256)

    # Discriminator network
    disc_hidden_dims: list = (1024, 1024, 512)

    # Encoder network
    enc_hidden_dims: list = (1024, 512)
    enc_separate: bool = False  # Share features with discriminator

    # AMP observation dimension (will be set by environment)
    amp_obs_dim: int = -1


@dataclass
class AWDPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """AWD PPO algorithm configuration."""

    # PPO parameters (from old config)
    value_loss_coef: float = 5.0  # critic_coef
    use_clipped_value_loss: bool = False  # clip_value
    clip_param: float = 0.2  # e_clip
    entropy_coef: float = 0.0
    num_learning_epochs: int = 6  # mini_epochs
    num_mini_batches: int = 8  # Calculated from minibatch_size=16384
    learning_rate: float = 2e-5
    schedule: str = "constant"  # lr_schedule
    gamma: float = 0.99
    lam: float = 0.95  # tau (GAE lambda)
    max_grad_norm: float = 1.0  # grad_norm
    desired_kl: float = 0.01

    # AWD-specific parameters
    # Latent management
    latent_dim: int = 64
    latent_steps_min: int = 1
    latent_steps_max: int = 150

    # Discriminator
    disc_coef: float = 5.0
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 10.0
    disc_weight_decay: float = 0.0001
    disc_reward_scale: float = 2.0

    # Encoder
    enc_coef: float = 5.0
    enc_weight_decay: float = 0.0
    enc_reward_scale: float = 1.0
    enc_grad_penalty: float = 0.0

    # Diversity bonus
    amp_diversity_bonus: float = 0.01
    amp_diversity_tar: float = 1.0

    # Reward weights
    task_reward_w: float = 0.0
    disc_reward_w: float = 0.5
    enc_reward_w: float = 0.5

    # AMP replay buffers
    amp_obs_demo_buffer_size: int = 80000
    amp_replay_buffer_size: int = 80000
    amp_replay_keep_prob: float = 0.01
    amp_batch_size: int = 512
    amp_minibatch_size: int = 4096

    # Other
    enable_eps_greedy: bool = False
    normalize_amp_input: bool = True


@dataclass
class AWDPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """AWD PPO runner configuration."""

    # Training parameters
    num_steps_per_env: int = 32  # horizon_length
    max_iterations: int = 100000  # max_epochs
    save_interval: int = 1000  # save_frequency

    # Experiment naming
    experiment_name: str = "awd_duckling"
    run_name: str = ""

    # Device
    device: str = "cuda:0"
    seed: int = 42

    # Resume
    resume: bool = False
    load_run: int = -1
    load_checkpoint: str = "model"

    # Logging
    logger: str = "tensorboard"
    log_interval: int = 1

    # Policy and algorithm configs (use field with default_factory for mutable defaults)
    policy: AWDPPOActorCriticCfg = field(default_factory=AWDPPOActorCriticCfg)
    algorithm: AWDPPOAlgorithmCfg = field(default_factory=AWDPPOAlgorithmCfg)

    def to_dict(self) -> dict:
        """Convert config to dictionary for RSL-RL runner.

        Returns:
            Configuration dictionary.
        """
        return {
            "seed": self.seed,
            "device": self.device,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "empirical_normalization": False,
            "policy": {
                "class_name": "AWDActorCritic",
                "init_noise_std": self.policy.init_noise_std,
                "actor_hidden_dims": list(self.policy.actor_hidden_dims),
                "critic_hidden_dims": list(self.policy.critic_hidden_dims),
                "activation": self.policy.activation,
                "latent_dim": self.policy.latent_dim,
                "style_dim": self.policy.style_dim,
                "style_hidden_dims": list(self.policy.style_hidden_dims),
                "disc_hidden_dims": list(self.policy.disc_hidden_dims),
                "enc_hidden_dims": list(self.policy.enc_hidden_dims),
                "enc_separate": self.policy.enc_separate,
            },
            "algorithm": {
                "class_name": "AWDPPO",
                "value_loss_coef": self.algorithm.value_loss_coef,
                "use_clipped_value_loss": self.algorithm.use_clipped_value_loss,
                "clip_param": self.algorithm.clip_param,
                "entropy_coef": self.algorithm.entropy_coef,
                "num_learning_epochs": self.algorithm.num_learning_epochs,
                "num_mini_batches": self.algorithm.num_mini_batches,
                "learning_rate": self.algorithm.learning_rate,
                "schedule": self.algorithm.schedule,
                "gamma": self.algorithm.gamma,
                "lam": self.algorithm.lam,
                "max_grad_norm": self.algorithm.max_grad_norm,
                "desired_kl": self.algorithm.desired_kl,
                # AWD-specific
                "latent_dim": self.algorithm.latent_dim,
                "latent_steps_min": self.algorithm.latent_steps_min,
                "latent_steps_max": self.algorithm.latent_steps_max,
                "disc_coef": self.algorithm.disc_coef,
                "disc_logit_reg": self.algorithm.disc_logit_reg,
                "disc_grad_penalty": self.algorithm.disc_grad_penalty,
                "disc_weight_decay": self.algorithm.disc_weight_decay,
                "disc_reward_scale": self.algorithm.disc_reward_scale,
                "enc_coef": self.algorithm.enc_coef,
                "enc_weight_decay": self.algorithm.enc_weight_decay,
                "enc_reward_scale": self.algorithm.enc_reward_scale,
                "enc_grad_penalty": self.algorithm.enc_grad_penalty,
                "amp_diversity_bonus": self.algorithm.amp_diversity_bonus,
                "amp_diversity_tar": self.algorithm.amp_diversity_tar,
                "task_reward_w": self.algorithm.task_reward_w,
                "disc_reward_w": self.algorithm.disc_reward_w,
                "enc_reward_w": self.algorithm.enc_reward_w,
                "amp_obs_demo_buffer_size": self.algorithm.amp_obs_demo_buffer_size,
                "amp_replay_buffer_size": self.algorithm.amp_replay_buffer_size,
                "amp_replay_keep_prob": self.algorithm.amp_replay_keep_prob,
                "amp_batch_size": self.algorithm.amp_batch_size,
                "amp_minibatch_size": self.algorithm.amp_minibatch_size,
                "enable_eps_greedy": self.algorithm.enable_eps_greedy,
                "normalize_amp_input": self.algorithm.normalize_amp_input,
            },
        }


# Task-specific configurations
@dataclass
class DucklingCommandAWDRunnerCfg(AWDPPORunnerCfg):
    """AWD configuration for DucklingCommand task."""

    experiment_name: str = "DucklingCommand_AWD"


@dataclass
class DucklingHeadingAWDRunnerCfg(AWDPPORunnerCfg):
    """AWD configuration for DucklingHeading task."""

    experiment_name: str = "DucklingHeading_AWD"


@dataclass
class DucklingPerturbAWDRunnerCfg(AWDPPORunnerCfg):
    """AWD configuration for DucklingPerturb task."""

    experiment_name: str = "DucklingPerturb_AWD"
