"""AWD (AMP with Diversity) PPO Algorithm for RSL-RL.

This is a port of the AWD algorithm from old_awd/learning/awd_agent.py to RSL-RL.
AWD extends AMP by adding an encoder network for style/diversity representation.

Key components:
- Discriminator: Learns to distinguish between agent and demo motions (from AMP)
- Encoder: Learns to predict latent style codes from observations
- Latent codes: Random style vectors that condition the policy for diverse behaviors
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class RunningMeanStd:
    """Running mean and standard deviation tracker for normalization."""

    def __init__(self, shape=(1,), device="cpu"):
        """Initialize running statistics.

        Args:
            shape: Shape of the data to track.
            device: Device to store tensors on.
        """
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(0, device=device, dtype=torch.float32)

    def update(self, x: torch.Tensor):
        """Update running statistics with new data.

        Args:
            x: New data batch [batch_size, *shape].
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Normalize data using running statistics.

        Args:
            x: Data to normalize [batch_size, *shape].
            epsilon: Small constant for numerical stability.

        Returns:
            Normalized data.
        """
        return (x - self.mean) / torch.sqrt(self.var + epsilon)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow calling the object directly for normalization.

        Args:
            x: Data to normalize.

        Returns:
            Normalized data.
        """
        return self.normalize(x)


class AWDPPO(PPO):
    """AWD PPO algorithm with discriminator and encoder networks."""

    def __init__(
        self,
        actor_critic: ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        # AWD-specific parameters
        latent_dim: int = 64,
        latent_steps_min: int = 150,
        latent_steps_max: int = 300,
        disc_coef: float = 5.0,
        disc_logit_reg: float = 0.05,
        disc_grad_penalty: float = 5.0,
        disc_weight_decay: float = 0.0001,
        disc_reward_scale: float = 2.0,
        enc_coef: float = 5.0,
        enc_weight_decay: float = 0.0001,
        enc_reward_scale: float = 1.0,
        enc_grad_penalty: float = 5.0,
        amp_diversity_bonus: float = 0.0,
        amp_diversity_tar: float = 1.5,
        task_reward_w: float = 0.0,
        disc_reward_w: float = 0.5,
        enc_reward_w: float = 0.5,
        amp_batch_size: int = 512,
        amp_minibatch_size: int = 4096,
        enable_eps_greedy: bool = True,
        normalize_amp_input: bool = True,
    ):
        """Initialize AWD PPO.

        Args:
            actor_critic: Actor-critic network with discriminator and encoder.
            num_learning_epochs: Number of epochs per update.
            num_mini_batches: Number of mini-batches per epoch.
            clip_param: PPO clipping parameter.
            gamma: Discount factor.
            lam: GAE lambda.
            value_loss_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            learning_rate: Learning rate.
            max_grad_norm: Max gradient norm for clipping.
            use_clipped_value_loss: Whether to clip value loss.
            schedule: Learning rate schedule type.
            desired_kl: Target KL divergence for adaptive schedule.
            device: Device to run on.
            latent_dim: Dimension of latent style codes.
            latent_steps_min: Minimum steps before changing latent.
            latent_steps_max: Maximum steps before changing latent.
            disc_coef: Discriminator loss coefficient.
            disc_logit_reg: Discriminator logit regularization.
            disc_grad_penalty: Discriminator gradient penalty.
            disc_weight_decay: Discriminator weight decay.
            disc_reward_scale: Discriminator reward scale.
            enc_coef: Encoder loss coefficient.
            enc_weight_decay: Encoder weight decay.
            enc_reward_scale: Encoder reward scale.
            enc_grad_penalty: Encoder gradient penalty.
            amp_diversity_bonus: Diversity bonus coefficient.
            amp_diversity_tar: Target diversity value.
            task_reward_w: Task reward weight in combined reward.
            disc_reward_w: Discriminator reward weight.
            enc_reward_w: Encoder reward weight.
            amp_batch_size: Batch size for AMP demo samples.
            amp_minibatch_size: Mini-batch size for discriminator.
            enable_eps_greedy: Enable epsilon-greedy action selection.
            normalize_amp_input: Normalize AMP observations.
        """
        super().__init__(
            policy=actor_critic,  # RSL-RL calls it 'policy', not 'actor_critic'
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
        )

        # AWD-specific parameters
        self.latent_dim = latent_dim
        self.latent_steps_min = latent_steps_min
        self.latent_steps_max = latent_steps_max
        self.disc_coef = disc_coef
        self.disc_logit_reg = disc_logit_reg
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_weight_decay = disc_weight_decay
        self.disc_reward_scale = disc_reward_scale
        self.enc_coef = enc_coef
        self.enc_weight_decay = enc_weight_decay
        self.enc_reward_scale = enc_reward_scale
        self.enc_grad_penalty = enc_grad_penalty
        self.amp_diversity_bonus = amp_diversity_bonus
        self.amp_diversity_tar = amp_diversity_tar
        self.task_reward_w = task_reward_w
        self.disc_reward_w = disc_reward_w
        self.enc_reward_w = enc_reward_w
        self.amp_batch_size = amp_batch_size
        self.amp_minibatch_size = amp_minibatch_size
        self.enable_eps_greedy = enable_eps_greedy
        self.normalize_amp_input = normalize_amp_input

        # Latent code management
        self.current_latents = None
        self.latent_reset_steps = None

        # AMP observation normalization
        if self.normalize_amp_input:
            self.amp_normalizer = RunningMeanStd(shape=(1,), device=device)  # Will be resized when we know AMP obs size

        # Replay buffers (to be initialized when we know sizes)
        self.amp_demo_buffer = None
        self.amp_replay_buffer = None

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs,  # TensorDict
        actions_shape,
    ):
        """Initialize storage including AWD-specific buffers.

        Args:
            training_type: Type of training (e.g., "on_policy").
            num_envs: Number of parallel environments.
            num_transitions_per_env: Number of transitions to store per environment.
            obs: TensorDict containing observation tensors.
            actions_shape: Shape of actions.
        """
        # Initialize parent storage with new API
        super().init_storage(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
        )

        # Initialize latent codes
        self.current_latents = torch.zeros(
            (num_envs, self.latent_dim), device=self.device, dtype=torch.float32
        )
        self.latent_reset_steps = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self._reset_all_latents(torch.arange(num_envs, device=self.device))

        # Build epsilon-greedy probabilities
        if self.enable_eps_greedy:
            env_ids = torch.arange(num_envs, device=self.device, dtype=torch.float32)
            self.rand_action_probs = 1.0 - torch.exp(10 * (env_ids / (num_envs - 1.0) - 1.0))
            self.rand_action_probs[0] = 1.0
            self.rand_action_probs[-1] = 0.0
        else:
            self.rand_action_probs = torch.ones(num_envs, device=self.device)

    def _reset_all_latents(self, env_ids: torch.Tensor):
        """Reset latent codes for specified environments.

        Args:
            env_ids: Indices of environments to reset.
        """
        n = len(env_ids)
        # Sample random latents from unit sphere
        z = torch.normal(0, 1, size=(n, self.latent_dim), device=self.device)
        z = torch.nn.functional.normalize(z, dim=-1)
        self.current_latents[env_ids] = z

        # Reset latent step counters
        self.latent_reset_steps[env_ids] = torch.randint(
            self.latent_steps_min,
            self.latent_steps_max,
            (n,),
            device=self.device,
            dtype=torch.int32,
        )

    def _update_latents(self, progress_buf: torch.Tensor):
        """Update latent codes based on progress.

        Args:
            progress_buf: Current progress/step count for each environment.
        """
        new_latent_envs = self.latent_reset_steps <= progress_buf
        need_update = torch.any(new_latent_envs)

        if need_update:
            new_latent_env_ids = new_latent_envs.nonzero(as_tuple=False).flatten()
            self._reset_all_latents(new_latent_env_ids)
            self.latent_reset_steps[new_latent_env_ids] += torch.randint(
                self.latent_steps_min,
                self.latent_steps_max,
                (len(new_latent_env_ids),),
                device=self.device,
                dtype=torch.int32,
            )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, progress_buf: torch.Tensor):
        """Get actions from policy with epsilon-greedy exploration.

        Args:
            obs: Observations.
            critic_obs: Critic observations.
            progress_buf: Progress buffer for latent updates.

        Returns:
            Tuple of (actions, values, log_probs, latents).
        """
        # Update latents if needed
        self._update_latents(progress_buf)

        # Get action from actor-critic (with latents)
        with torch.no_grad():
            # Pass latents to actor-critic
            actions, _, log_probs, _ = self.policy.act_inference(obs, latents=self.current_latents)
            # Sum log_probs over action dimensions for PPO loss
            log_probs_sum = log_probs.sum(dim=-1, keepdim=True)
            # evaluate() returns (log_prob, entropy, value, mu, sigma) - we only need value
            _, _, values, _, _ = self.policy.evaluate(critic_obs, obs, actions, self.current_latents)

        # Apply epsilon-greedy: replace some actions with deterministic (mean) actions
        rand_action_mask = torch.bernoulli(self.rand_action_probs.unsqueeze(-1))
        # Note: RSL-RL's actor returns sampled actions, we'd need the mean separately
        # For now, we'll skip this and just use sampled actions
        # In full implementation, we'd get both mu and sampled actions

        # Return log_probs per dimension for storage (RSL-RL expects this)
        return actions, values, log_probs, self.current_latents.clone()

    def compute_returns(self, last_values: torch.Tensor):
        """Compute returns with AMP rewards.

        This is called after rollout collection to compute advantages.

        Args:
            last_values: Values for the last step (bootstrap).
        """
        # First compute discriminator and encoder rewards
        # Note: This requires amp_obs to be stored in rollout buffer
        # We'll need to extend RolloutStorage for this

        # For now, use parent implementation
        # In full version, we'd:
        # 1. Compute disc_rewards from amp_obs
        # 2. Compute enc_rewards from amp_obs and latents
        # 3. Combine: rewards = task_w * task_r + disc_w * disc_r + enc_w * enc_r
        # 4. Then compute advantages with combined rewards

        super().compute_returns(last_values)

    def update(self):
        """Update policy, value, discriminator, and encoder networks.

        Returns:
            Dictionary with training statistics.
        """
        # Get parent PPO statistics
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_disc_loss = 0
        mean_enc_loss = 0
        mean_disc_agent_acc = 0
        mean_disc_demo_acc = 0
        mean_enc_grad_penalty = 0
        mean_diversity_loss = 0

        # Prepare generator for mini-batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # In full implementation, we'd also get:
            # - amp_obs_batch
            # - latents_batch
            # - rand_action_mask_batch

            # PPO update (parent class logic)
            actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = (
                self.policy.evaluate(
                    critic_obs_batch,
                    obs_batch,
                    actions_batch,
                )
            )

            # PPO actor loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # PPO value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Total loss (will add disc and enc losses)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # TODO: Add discriminator loss
            # disc_loss = self._compute_disc_loss(amp_obs_batch, amp_obs_demo, amp_obs_replay)
            # loss += self.disc_coef * disc_loss

            # TODO: Add encoder loss
            # enc_loss = self._compute_enc_loss(amp_obs_batch, latents_batch, rand_action_mask)
            # loss += self.enc_coef * enc_loss

            # TODO: Add diversity loss if enabled
            # if self.amp_diversity_bonus > 0:
            #     div_loss = self._compute_diversity_loss(obs_batch, mu_batch, latents_batch, rand_action_mask)
            #     loss += self.amp_diversity_bonus * div_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Accumulate stats
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        # Average over all mini-batches
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "disc_loss": mean_disc_loss,
            "enc_loss": mean_enc_loss,
            "disc_agent_acc": mean_disc_agent_acc,
            "disc_demo_acc": mean_disc_demo_acc,
            "enc_grad_penalty": mean_enc_grad_penalty,
            "diversity_loss": mean_diversity_loss,
        }

    def _compute_disc_loss(
        self,
        amp_obs_agent: torch.Tensor,
        amp_obs_demo: torch.Tensor,
        amp_obs_replay: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator loss.

        Args:
            amp_obs_agent: AMP observations from current policy.
            amp_obs_demo: AMP observations from demonstrations.
            amp_obs_replay: AMP observations from replay buffer.

        Returns:
            Discriminator loss.
        """
        # Normalize inputs if needed
        if self.normalize_amp_input:
            amp_obs_agent = self.amp_normalizer(amp_obs_agent)
            amp_obs_demo = self.amp_normalizer(amp_obs_demo)
            amp_obs_replay = self.amp_normalizer(amp_obs_replay)

        # Enable gradient computation for demo observations (for gradient penalty)
        amp_obs_demo.requires_grad_(True)

        # Get discriminator predictions
        disc_agent_logit = self.policy.discriminator(amp_obs_agent)
        disc_replay_logit = self.policy.discriminator(amp_obs_replay)
        disc_demo_logit = self.policy.discriminator(amp_obs_demo)

        # Combine agent and replay
        disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_replay_logit], dim=0)

        # BCE loss: agent/replay should be classified as 0, demo as 1
        bce = nn.BCEWithLogitsLoss()
        disc_loss_agent = bce(disc_agent_cat_logit, torch.zeros_like(disc_agent_cat_logit))
        disc_loss_demo = bce(disc_demo_logit, torch.ones_like(disc_demo_logit))
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # Logit regularization
        if self.disc_logit_reg > 0:
            logit_weights = self.policy.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self.disc_logit_reg * disc_logit_loss

        # Gradient penalty
        if self.disc_grad_penalty > 0:
            disc_demo_grad = torch.autograd.grad(
                disc_demo_logit,
                amp_obs_demo,
                grad_outputs=torch.ones_like(disc_demo_logit),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
            disc_grad_penalty = torch.mean(disc_demo_grad)
            disc_loss += self.disc_grad_penalty * disc_grad_penalty

        # Weight decay
        if self.disc_weight_decay > 0:
            disc_weights = self.policy.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self.disc_weight_decay * disc_weight_decay

        return disc_loss

    def _compute_enc_loss(
        self, amp_obs: torch.Tensor, latents: torch.Tensor, loss_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute encoder loss.

        Args:
            amp_obs: AMP observations.
            latents: Target latent codes.
            loss_mask: Mask for valid samples.

        Returns:
            Encoder loss.
        """
        # Normalize inputs if needed
        if self.normalize_amp_input:
            amp_obs = self.amp_normalizer(amp_obs)

        # Enable gradients for gradient penalty
        if self.enc_grad_penalty > 0:
            amp_obs.requires_grad_(True)

        # Get encoder predictions
        enc_pred = self.policy.encoder(amp_obs)

        # Compute error (negative dot product, want predictions to match latents)
        enc_err = enc_pred * latents
        enc_err = -torch.sum(enc_err, dim=-1, keepdim=True)
        enc_loss = torch.mean(enc_err)

        # Weight decay
        if self.enc_weight_decay > 0:
            enc_weights = self.policy.get_enc_weights()
            enc_weights = torch.cat(enc_weights, dim=-1)
            enc_weight_decay = torch.sum(torch.square(enc_weights))
            enc_loss += self.enc_weight_decay * enc_weight_decay

        # Gradient penalty
        if self.enc_grad_penalty > 0:
            enc_obs_grad = torch.autograd.grad(
                enc_err,
                amp_obs,
                grad_outputs=torch.ones_like(enc_err),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            enc_obs_grad = torch.sum(torch.square(enc_obs_grad), dim=-1)
            enc_grad_penalty = torch.mean(enc_obs_grad)
            enc_loss += self.enc_grad_penalty * enc_grad_penalty

        return enc_loss

    def _compute_diversity_loss(
        self, obs: torch.Tensor, actions: torch.Tensor, latents: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute diversity bonus loss.

        Args:
            obs: Observations.
            actions: Action parameters (mu).
            latents: Current latent codes.
            mask: Mask for valid samples.

        Returns:
            Diversity loss.
        """
        n = obs.shape[0]

        # Sample new random latents
        new_z = torch.normal(0, 1, size=(n, self.latent_dim), device=self.device)
        new_z = torch.nn.functional.normalize(new_z, dim=-1)

        # Get actions with new latents
        with torch.no_grad():
            new_mu, _ = self.policy.actor(obs, latents=new_z)

        # Compute action difference
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        clipped_mu = torch.clamp(new_mu, -1.0, 1.0)
        a_diff = clipped_actions - clipped_mu
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        # Compute latent difference (normalized dot product distance)
        z_diff = new_z * latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        # Diversity bonus
        diversity_bonus = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(self.amp_diversity_tar - diversity_bonus)

        return diversity_loss
