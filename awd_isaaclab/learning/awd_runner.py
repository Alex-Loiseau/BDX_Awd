"""AWD OnPolicy Runner for RSL-RL.

This extends RSL-RL's OnPolicyRunner to support AWD training with:
- AMP observations
- Discriminator and encoder networks
- Replay buffers for demonstrations
- Latent code management
"""

import os
import time
import torch
import numpy as np
import gymnasium as gym
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

from .awd_storage import AWDRolloutStorage
from .awd_ppo import AWDPPO
from .awd_models import AWDActorCritic
from .amp_replay_buffer import AMPDemoBuffer, AMPReplayBuffer


class AWDOnPolicyRunner(OnPolicyRunner):
    """AWD runner for on-policy RL with AMP and diversity.

    Extends OnPolicyRunner to handle AWD-specific training.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str = None, device: str = "cpu"):
        """Initialize AWD runner.

        Args:
            env: Vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging.
            device: Device to run on.
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Get environment info
        # For DirectRLEnv, observations are provided as a dict with "policy" key
        # Get obs dimension from observation space
        obs_space = env.unwrapped.single_observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.num_obs = gym.spaces.flatdim(obs_space["policy"])
            # Check if there's a "critic" observation (privileged obs)
            if "critic" in obs_space.spaces:
                num_critic_obs = gym.spaces.flatdim(obs_space["critic"])
            else:
                num_critic_obs = self.num_obs
        else:
            self.num_obs = gym.spaces.flatdim(obs_space)
            num_critic_obs = self.num_obs

        # Get AMP observation dimension from environment
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "num_amp_obs"):
            self.num_amp_obs = env.unwrapped.num_amp_obs
        else:
            raise ValueError("Environment must provide num_amp_obs for AWD training")

        # Actor-critic network
        actor_critic_class = AWDActorCritic
        self.actor_critic = actor_critic_class(
            obs_dim=self.num_obs,
            action_dim=self.env.num_actions,
            amp_obs_dim=self.num_amp_obs,
            latent_dim=self.policy_cfg.get("latent_dim", 64),
            actor_hidden_dims=self.policy_cfg.get("actor_hidden_dims", [1024, 1024, 512]),
            critic_hidden_dims=self.policy_cfg.get("critic_hidden_dims", [1024, 1024, 512]),
            disc_hidden_dims=self.policy_cfg.get("disc_hidden_dims", [1024, 512]),
            enc_hidden_dims=self.policy_cfg.get("enc_hidden_dims", [1024, 512]),
            style_hidden_dims=self.policy_cfg.get("style_hidden_dims", [512, 256]),
            style_dim=self.policy_cfg.get("style_dim", 64),
            activation=self.policy_cfg.get("activation", "relu"),
            init_noise_std=self.policy_cfg.get("init_noise_std", 1.0),
            separate_encoder=self.policy_cfg.get("enc_separate", False),
        ).to(self.device)

        # AWD PPO algorithm
        alg_class = AWDPPO
        self.alg = alg_class(
            actor_critic=self.actor_critic,
            num_learning_epochs=self.alg_cfg["num_learning_epochs"],
            num_mini_batches=self.alg_cfg["num_mini_batches"],
            clip_param=self.alg_cfg["clip_param"],
            gamma=self.alg_cfg["gamma"],
            lam=self.alg_cfg["lam"],
            value_loss_coef=self.alg_cfg["value_loss_coef"],
            entropy_coef=self.alg_cfg["entropy_coef"],
            learning_rate=self.alg_cfg["learning_rate"],
            max_grad_norm=self.alg_cfg["max_grad_norm"],
            use_clipped_value_loss=self.alg_cfg["use_clipped_value_loss"],
            schedule=self.alg_cfg["schedule"],
            desired_kl=self.alg_cfg.get("desired_kl", 0.01),
            device=self.device,
            # AWD-specific
            latent_dim=self.alg_cfg.get("latent_dim", 64),
            latent_steps_min=self.alg_cfg.get("latent_steps_min", 150),
            latent_steps_max=self.alg_cfg.get("latent_steps_max", 300),
            disc_coef=self.alg_cfg.get("disc_coef", 5.0),
            disc_logit_reg=self.alg_cfg.get("disc_logit_reg", 0.05),
            disc_grad_penalty=self.alg_cfg.get("disc_grad_penalty", 5.0),
            disc_weight_decay=self.alg_cfg.get("disc_weight_decay", 0.0001),
            disc_reward_scale=self.alg_cfg.get("disc_reward_scale", 2.0),
            enc_coef=self.alg_cfg.get("enc_coef", 5.0),
            enc_weight_decay=self.alg_cfg.get("enc_weight_decay", 0.0001),
            enc_reward_scale=self.alg_cfg.get("enc_reward_scale", 1.0),
            enc_grad_penalty=self.alg_cfg.get("enc_grad_penalty", 5.0),
            amp_diversity_bonus=self.alg_cfg.get("amp_diversity_bonus", 0.0),
            amp_diversity_tar=self.alg_cfg.get("amp_diversity_tar", 1.5),
            task_reward_w=self.alg_cfg.get("task_reward_w", 0.0),
            disc_reward_w=self.alg_cfg.get("disc_reward_w", 0.5),
            enc_reward_w=self.alg_cfg.get("enc_reward_w", 0.5),
            amp_batch_size=self.alg_cfg.get("amp_batch_size", 512),
            amp_minibatch_size=self.alg_cfg.get("amp_minibatch_size", 4096),
            enable_eps_greedy=self.alg_cfg.get("enable_eps_greedy", True),
            normalize_amp_input=self.alg_cfg.get("normalize_amp_input", True),
        )

        # AWD Rollout storage
        self.storage = AWDRolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],
            obs_shape=[self.num_obs],
            privileged_obs_shape=[num_critic_obs],
            actions_shape=[self.env.num_actions],
            amp_obs_shape=[self.num_amp_obs],
            latent_shape=[self.alg_cfg.get("latent_dim", 64)],
            device=self.device,
        )

        # Initialize algorithm storage (new RSL-RL API)
        # Create TensorDict for observations
        from tensordict import TensorDict
        obs_dict = TensorDict(
            {
                "policy": torch.zeros(self.env.num_envs, self.num_obs, device=self.device),
                "critic": torch.zeros(self.env.num_envs, num_critic_obs, device=self.device),
            },
            batch_size=[self.env.num_envs],
        )

        self.alg.init_storage(
            training_type="on_policy",
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],
            obs=obs_dict,
            actions_shape=(self.env.num_actions,),
        )

        # Initialize AWD latents after algorithm creation
        self.alg.current_latents = torch.zeros(
            (self.env.num_envs, self.alg.latent_dim), device=self.device, dtype=torch.float32
        )
        self.alg.latent_reset_steps = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.int32)

        # AMP replay buffers
        self.amp_demo_buffer = AMPDemoBuffer(
            buffer_size=self.alg_cfg.get("amp_obs_demo_buffer_size", 80000),
            device=self.device,
        )

        self.amp_replay_buffer = AMPReplayBuffer(
            buffer_size=self.alg_cfg.get("amp_replay_buffer_size", 80000),
            device=self.device,
        )

        self.amp_replay_keep_prob = self.alg_cfg.get("amp_replay_keep_prob", 0.01)

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Run AWD training loop.

        Args:
            num_learning_iterations: Number of policy updates.
            init_at_random_ep_len: Whether to initialize at random episode lengths.
        """
        # Initialize
        # Get observations returns a TensorDict with "policy" and "critic" keys
        obs_dict = self.env.get_observations()
        obs = obs_dict["policy"].to(self.device)
        critic_obs = obs_dict.get("critic", obs_dict["policy"]).to(self.device)

        # Reset environment to get initial AMP observations
        self.env.reset()

        # Initialize demo buffer
        print("[AWD Runner] Initializing demonstration buffer...")
        self._init_amp_demo_buffer()

        # Training loop
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # Rollout
            with torch.inference_mode():
                for i in range(self.cfg["num_steps_per_env"]):
                    # Get progress buffer for latent updates
                    if hasattr(self.env.unwrapped, "episode_length_buf"):
                        progress_buf = self.env.unwrapped.episode_length_buf
                    else:
                        progress_buf = torch.zeros(
                            self.env.num_envs, dtype=torch.int, device=self.device
                        )

                    # Get actions from policy (with latent updates)
                    actions, values, log_probs, latents = self.alg.act(obs, critic_obs, progress_buf)

                    # Step environment
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs = obs_dict["policy"]
                    critic_obs = obs_dict.get("critic", obs_dict["policy"])

                    # Get AMP observations
                    if hasattr(infos, "get") and "amp_obs" in infos:
                        amp_obs = infos["amp_obs"]
                    elif hasattr(self.env.unwrapped, "get_amp_observations"):
                        amp_obs = self.env.unwrapped.get_amp_observations()
                    else:
                        raise ValueError("Environment must provide AMP observations")

                    # Get random action mask
                    rand_action_mask = torch.bernoulli(
                        self.alg.rand_action_probs.unsqueeze(-1)
                    )

                    # Compute AMP rewards
                    disc_rewards = self._compute_disc_rewards(amp_obs)
                    enc_rewards = self._compute_enc_rewards(amp_obs, latents)

                    # Combine rewards
                    task_rewards = rewards.clone()
                    combined_rewards = (
                        self.alg.task_reward_w * task_rewards
                        + self.alg.disc_reward_w * disc_rewards.squeeze(-1)
                        + self.alg.enc_reward_w * enc_rewards.squeeze(-1)
                    )


                    # Store transition
                    self.storage.add_transitions(
                        observations=obs,
                        privileged_observations=critic_obs,
                        actions=actions,
                        rewards=combined_rewards,
                        dones=dones,
                        values=values,
                        log_probs=log_probs,
                        mu=torch.zeros_like(actions),  # Will get from actor
                        sigma=torch.zeros_like(actions),
                        amp_obs=amp_obs,
                        latents=latents,
                        rand_action_mask=rand_action_mask,
                    )

                    # Update episode info
                    cur_reward_sum += rewards
                    cur_episode_length += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                # Store AMP reward components for logging
                # This is a simplified version - full version would store per-step
                self.storage.store_amp_rewards(
                    task_rewards=task_rewards.unsqueeze(0).repeat(self.cfg["num_steps_per_env"], 1),
                    disc_rewards=disc_rewards.unsqueeze(0).repeat(self.cfg["num_steps_per_env"], 1, 1),
                    enc_rewards=enc_rewards.unsqueeze(0).repeat(self.cfg["num_steps_per_env"], 1, 1),
                )

                # Bootstrap value
                last_values = self.alg.policy.evaluate(
                    critic_obs, obs, actions=torch.zeros_like(actions), latents=latents
                )[2]

            # Update
            stop = time.time()
            collection_time = stop - start

            # Compute returns
            self.storage.compute_returns(last_values, self.alg.gamma, self.alg.lam)

            # Update AMP buffers
            self._update_amp_buffers()

            # Policy update
            mean_value_loss, mean_surrogate_loss = self._update()

            # Logging
            self.tot_timesteps += self.cfg["num_steps_per_env"] * self.env.num_envs
            self.tot_time += collection_time

            if self.log_dir is not None:
                self.log(locals())

            # Save
            if it % self.cfg.get("save_interval", 50) == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            self.storage.clear()
            self.current_learning_iteration = it

        self.save(os.path.join(self.log_dir, "model.pt"))

    def _init_amp_demo_buffer(self):
        """Initialize demonstration buffer from environment."""
        if hasattr(self.env.unwrapped, "fetch_amp_obs_demo"):
            # Fill buffer with demonstrations
            batch_size = self.alg_cfg.get("amp_batch_size", 512)
            buffer_size = self.amp_demo_buffer.get_buffer_size()
            num_batches = int(np.ceil(buffer_size / batch_size))

            for i in range(min(num_batches, 10)):  # Limit initial fill
                amp_obs_demo = self.env.unwrapped.fetch_amp_obs_demo(batch_size)
                self.amp_demo_buffer.store({"amp_obs": amp_obs_demo})

            print(f"[AWD Runner] Loaded {len(self.amp_demo_buffer)} demonstration samples")
        else:
            print("[AWD Runner] Warning: No demonstration data available")

    def _compute_disc_rewards(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Compute discriminator rewards.

        Args:
            amp_obs: AMP observations.

        Returns:
            Discriminator rewards.
        """
        with torch.no_grad():
            # Normalize if needed
            if self.alg.normalize_amp_input and hasattr(self.alg, "amp_normalizer"):
                amp_obs_norm = self.alg.amp_normalizer(amp_obs)
            else:
                amp_obs_norm = amp_obs

            # Get discriminator logits
            disc_logits = self.actor_critic.discriminator(amp_obs_norm)

            # Compute reward: -log(1 - D(s))
            prob = 1.0 / (1.0 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1.0 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self.alg.disc_reward_scale

        return disc_r

    def _compute_enc_rewards(self, amp_obs: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Compute encoder rewards.

        Args:
            amp_obs: AMP observations.
            latents: Target latent codes.

        Returns:
            Encoder rewards.
        """
        with torch.no_grad():
            # Normalize if needed
            if self.alg.normalize_amp_input and hasattr(self.alg, "amp_normalizer"):
                amp_obs_norm = self.alg.amp_normalizer(amp_obs)
            else:
                amp_obs_norm = amp_obs

            # Get encoder predictions
            enc_pred = self.actor_critic.encoder(amp_obs_norm)

            # Compute error
            err = enc_pred * latents
            err = -torch.sum(err, dim=-1, keepdim=True)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self.alg.enc_reward_scale

        return enc_r

    def _update_amp_buffers(self):
        """Update AMP demonstration and replay buffers."""
        # Update demo buffer
        if hasattr(self.env.unwrapped, "fetch_amp_obs_demo"):
            batch_size = self.alg_cfg.get("amp_batch_size", 512)
            amp_obs_demo = self.env.unwrapped.fetch_amp_obs_demo(batch_size)
            self.amp_demo_buffer.store({"amp_obs": amp_obs_demo})

        # Update replay buffer with agent observations
        amp_obs = self.storage.amp_obs.flatten(0, 1)  # Flatten time and env dims

        # Subsample based on keep probability
        buf_size = self.amp_replay_buffer.get_buffer_size()
        buf_total_count = self.amp_replay_buffer.get_total_count()

        if buf_total_count > buf_size:
            keep_probs = torch.full(
                (amp_obs.shape[0],), self.amp_replay_keep_prob, device=self.device
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if amp_obs.shape[0] > buf_size:
            rand_idx = torch.randperm(amp_obs.shape[0])[:buf_size]
            amp_obs = amp_obs[rand_idx]

        if amp_obs.shape[0] > 0:
            self.amp_replay_buffer.store({"amp_obs": amp_obs})

    def _update(self):
        """Perform policy update."""
        # This is simplified - full implementation would handle all AWD losses
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        # Use custom mini-batch generator with AWD data
        generator = self.storage.mini_batch_generator(
            self.alg.num_mini_batches, self.alg.num_learning_epochs
        )

        for mini_batch in generator:
            # Unpack mini-batch (includes AWD-specific tensors)
            # Full update would call self.alg.update() with discriminator/encoder losses
            pass

        return mean_value_loss, mean_surrogate_loss

    def log(self, locs: dict):
        """Log training statistics."""
        if self.writer is None:
            return

        it = locs["it"]
        rewbuffer = locs["rewbuffer"]
        lenbuffer = locs["lenbuffer"]

        # Episode info
        if len(rewbuffer) > 0:
            self.writer.add_scalar("Episode/mean_reward", np.mean(rewbuffer), it)
            self.writer.add_scalar("Episode/mean_length", np.mean(lenbuffer), it)

        # Storage statistics
        stats = self.storage.get_statistics()
        for key, value in stats.items():
            self.writer.add_scalar(f"Storage/{key}", value, it)

    def save(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iteration": self.current_learning_iteration,
                "infos": {},
            },
            path,
        )
