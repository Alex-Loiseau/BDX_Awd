"""Extended Rollout Storage for AWD/AMP training.

This extends RSL-RL's RolloutStorage to include:
- AMP observations
- Latent codes
- Random action masks for epsilon-greedy
- Discriminator and encoder rewards
"""

import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage


class AWDRolloutStorage(RolloutStorage):
    """Extended rollout storage for AWD/AMP.

    Adds storage for AMP-specific data on top of standard PPO rollouts.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: list,
        privileged_obs_shape: list,
        actions_shape: list,
        amp_obs_shape: list,
        latent_shape: list,
        device: str = "cpu",
    ):
        """Initialize AWD rollout storage.

        Args:
            num_envs: Number of parallel environments.
            num_transitions_per_env: Number of transitions per environment.
            obs_shape: Shape of observations.
            privileged_obs_shape: Shape of privileged observations.
            actions_shape: Shape of actions.
            amp_obs_shape: Shape of AMP observations.
            latent_shape: Shape of latent codes.
            device: Device to store tensors on.
        """
        # Create TensorDict for observations
        # RSL-RL's new API expects obs as a TensorDict with "policy" and optionally "critic" keys
        obs_dict = TensorDict(
            {
                "policy": torch.zeros(num_envs, *obs_shape, device=device),
                "critic": torch.zeros(num_envs, *privileged_obs_shape, device=device),
            },
            batch_size=[num_envs],
        )

        # Initialize parent storage with new API
        # training_type must be "rl" for reinforcement learning (PPO)
        super().__init__(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs_dict,
            actions_shape=tuple(actions_shape),
            device=device,
        )

        # AMP observation storage
        self.amp_obs = torch.zeros(
            num_transitions_per_env, num_envs, *amp_obs_shape, device=self.device
        )

        # Latent code storage
        self.latents = torch.zeros(
            num_transitions_per_env, num_envs, *latent_shape, device=self.device
        )

        # Random action mask for epsilon-greedy
        self.rand_action_mask = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        # Discriminator rewards
        self.disc_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        # Encoder rewards
        self.enc_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        # Task rewards (original env rewards, before combination)
        self.task_rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )

        # Standard PPO tensors - parent might not create these in new API
        # so we create them here to be safe
        if not hasattr(self, 'values'):
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        if not hasattr(self, 'actions_log_prob'):
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        if not hasattr(self, 'mu'):
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        if not hasattr(self, 'sigma'):
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

    def add_transitions(
        self,
        observations: torch.Tensor,
        privileged_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        amp_obs: torch.Tensor,
        latents: torch.Tensor,
        rand_action_mask: torch.Tensor,
    ):
        """Add transition to storage.

        Args:
            observations: Observations.
            privileged_observations: Privileged observations.
            actions: Actions.
            rewards: Rewards (already combined task + disc + enc).
            dones: Done flags.
            values: Value estimates.
            log_probs: Action log probabilities.
            mu: Action means.
            sigma: Action standard deviations.
            amp_obs: AMP observations.
            latents: Latent codes.
            rand_action_mask: Random action mask.
        """
        # Store standard PPO data
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        # Handle observations: self.observations is a TensorDict but observations is a tensor
        self.observations[self.step]["policy"].copy_(observations)
        # Privileged observations go into the "critic" key
        if privileged_observations is not None:
            self.observations[self.step]["critic"].copy_(privileged_observations)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        # RSL-RL expects log_probs as a scalar (summed over action dimensions)
        if log_probs.dim() == 2 and log_probs.shape[1] > 1:
            # Sum over action dimensions if needed
            log_probs_scalar = log_probs.sum(dim=-1, keepdim=True)
            self.actions_log_prob[self.step].copy_(log_probs_scalar)
        else:
            self.actions_log_prob[self.step].copy_(log_probs.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        # Store AWD-specific data
        self.amp_obs[self.step].copy_(amp_obs)
        self.latents[self.step].copy_(latents)
        self.rand_action_mask[self.step].copy_(rand_action_mask.view(-1, 1))

        self.step += 1

    def store_amp_rewards(
        self,
        task_rewards: torch.Tensor,
        disc_rewards: torch.Tensor,
        enc_rewards: torch.Tensor,
    ):
        """Store separate reward components for logging.

        This is called after rollout collection to decompose the combined rewards.

        Args:
            task_rewards: Task rewards from environment.
            disc_rewards: Discriminator rewards.
            enc_rewards: Encoder rewards.
        """
        # Store all transitions
        for i in range(self.num_transitions_per_env):
            self.task_rewards[i].copy_(task_rewards[i].view(-1, 1))
            self.disc_rewards[i].copy_(disc_rewards[i].view(-1, 1))
            self.enc_rewards[i].copy_(enc_rewards[i].view(-1, 1))

    def get_statistics(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with statistics.
        """
        # Create statistics dictionary
        stats = {}

        # Add AWD-specific statistics
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)

        # Task rewards
        task_rewards = self.task_rewards.cpu()
        flat_task_rewards = task_rewards.permute(1, 0, 2).reshape(-1, 1)
        stats["task_reward_mean"] = torch.mean(flat_task_rewards).item()
        stats["task_reward_std"] = torch.std(flat_task_rewards).item()

        # Discriminator rewards
        disc_rewards = self.disc_rewards.cpu()
        flat_disc_rewards = disc_rewards.permute(1, 0, 2).reshape(-1, 1)
        stats["disc_reward_mean"] = torch.mean(flat_disc_rewards).item()
        stats["disc_reward_std"] = torch.std(flat_disc_rewards).item()

        # Encoder rewards
        enc_rewards = self.enc_rewards.cpu()
        flat_enc_rewards = enc_rewards.permute(1, 0, 2).reshape(-1, 1)
        stats["enc_reward_mean"] = torch.mean(flat_enc_rewards).item()
        stats["enc_reward_std"] = torch.std(flat_enc_rewards).item()

        return stats

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Generate mini-batches for training.

        Args:
            num_mini_batches: Number of mini-batches.
            num_epochs: Number of epochs.

        Yields:
            Tuples of mini-batch data including AWD-specific tensors.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # Prepare data (flatten time and env dimensions)
        # observations is a TensorDict, extract the tensors
        observations = self.observations["policy"].flatten(0, 1)
        critic_observations = self.observations["critic"].flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # AWD-specific data
        amp_obs = self.amp_obs.flatten(0, 1)
        latents = self.latents.flatten(0, 1)
        rand_action_mask = self.rand_action_mask.flatten(0, 1)

        for epoch in range(num_epochs):
            # Generate random permutation
            perm = torch.randperm(batch_size, device=self.device)

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = perm[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # AWD-specific batches
                amp_obs_batch = amp_obs[batch_idx]
                latents_batch = latents[batch_idx]
                rand_action_mask_batch = rand_action_mask[batch_idx]

                yield (
                    obs_batch,
                    critic_observations_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    amp_obs_batch,
                    latents_batch,
                    rand_action_mask_batch,
                )
