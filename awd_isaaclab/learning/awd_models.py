"""AWD Network Models for RSL-RL.

This module contains the neural network architectures for AWD (AMP with Diversity):
- Discriminator: Distinguishes between agent and demo motions
- Encoder: Predicts latent style codes from observations
- Style-conditioned Actor-Critic: Generates diverse behaviors based on latent codes

Ported from old_awd/learning/awd_network_builder.py and amp_network_builder.py
"""

import torch
import torch.nn as nn
from typing import Tuple


# Initialization scales
DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1


class AMPDiscriminator(nn.Module):
    """Discriminator network for AMP.

    Learns to distinguish between agent motions and demonstration motions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [1024, 512],
        activation: str = "relu",
    ):
        """Initialize discriminator.

        Args:
            input_dim: Dimension of AMP observations.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build MLP
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.logits = nn.Linear(hidden_dims[-1], 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Logits layer with special initialization
        nn.init.uniform_(
            self.logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE
        )
        nn.init.zeros_(self.logits.bias)

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            amp_obs: AMP observations [batch, input_dim].

        Returns:
            Discriminator logits [batch, 1].
        """
        h = self.mlp(amp_obs)
        logits = self.logits(h)
        return logits

    def get_logit_weights(self) -> torch.Tensor:
        """Get flattened logit layer weights.

        Returns:
            Flattened weight tensor.
        """
        return torch.flatten(self.logits.weight)

    def get_all_weights(self) -> list:
        """Get all network weights as list of flattened tensors.

        Returns:
            List of flattened weight tensors.
        """
        weights = []
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        weights.append(torch.flatten(self.logits.weight))
        return weights


class AWDEncoder(nn.Module):
    """Encoder network for AWD.

    Predicts latent style codes from AMP observations.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list = [1024, 512],
        activation: str = "relu",
        separate_network: bool = False,
        disc_network: AMPDiscriminator = None,
    ):
        """Initialize encoder.

        Args:
            input_dim: Dimension of AMP observations.
            latent_dim: Dimension of latent codes.
            hidden_dims: List of hidden layer dimensions (used if separate_network=True).
            activation: Activation function name.
            separate_network: If True, use separate MLP. If False, share with discriminator.
            disc_network: Discriminator network to share features with (if separate_network=False).
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.separate_network = separate_network

        if separate_network:
            # Build separate MLP
            layers = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "elu":
                    layers.append(nn.ELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                in_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)
            mlp_out_dim = hidden_dims[-1]

            # Initialize weights
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            # Share features with discriminator
            assert disc_network is not None, "Must provide discriminator network to share"
            self.mlp = disc_network.mlp
            mlp_out_dim = disc_network.hidden_dims[-1]

        # Output layer
        self.output = nn.Linear(mlp_out_dim, latent_dim)

        # Initialize output layer
        nn.init.uniform_(self.output.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)
        nn.init.zeros_(self.output.bias)

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            amp_obs: AMP observations [batch, input_dim].

        Returns:
            Normalized latent predictions [batch, latent_dim].
        """
        h = self.mlp(amp_obs)
        output = self.output(h)
        # Normalize to unit sphere
        output = torch.nn.functional.normalize(output, dim=-1)
        return output

    def get_weights(self) -> list:
        """Get all network weights as list of flattened tensors.

        Returns:
            List of flattened weight tensors.
        """
        weights = []
        if self.separate_network:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))
        weights.append(torch.flatten(self.output.weight))
        return weights


class StyleMLP(nn.Module):
    """Style processing network.

    Transforms latent codes into style vectors for actor conditioning.
    """

    def __init__(
        self,
        latent_dim: int,
        style_dim: int,
        hidden_dims: list = [512, 256],
        activation: str = "relu",
    ):
        """Initialize style MLP.

        Args:
            latent_dim: Dimension of input latent codes.
            style_dim: Dimension of output style vectors.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
        """
        super().__init__()

        layers = []
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], style_dim)

        # Initialize
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output layer with special init
        nn.init.uniform_(self.output.weight, -1.0, 1.0)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            latent: Latent codes [batch, latent_dim].

        Returns:
            Style vectors [batch, style_dim].
        """
        h = self.mlp(latent)
        style = self.output(h)
        style = torch.tanh(style)  # Bound output
        return style


class StyleConditionedMLP(nn.Module):
    """Style-conditioned MLP for actor network.

    Concatenates observations with style vectors and processes through MLP.
    Based on AMPStyleCatNet1 from old code.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list = [1024, 1024, 512],
        style_hidden_dims: list = [512, 256],
        style_dim: int = 64,
        activation: str = "relu",
    ):
        """Initialize style-conditioned MLP.

        Args:
            obs_dim: Dimension of observations.
            latent_dim: Dimension of latent codes.
            output_dim: Dimension of output.
            hidden_dims: List of hidden layer dimensions for main MLP.
            style_hidden_dims: List of hidden layer dimensions for style processing.
            style_dim: Dimension of style vectors.
            activation: Activation function name.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.style_dim = style_dim

        # Style processing network
        self.style_net = StyleMLP(latent_dim, style_dim, style_hidden_dims, activation)

        # Main MLP (takes obs + style)
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "elu":
            act_fn = nn.ELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()

        layers = []
        in_dim = obs_dim + style_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim

        self.mlp = nn.ModuleList(layers)

        # Initialize
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observations [batch, obs_dim].
            latent: Latent codes [batch, latent_dim].

        Returns:
            Output features [batch, output_dim].
        """
        # Process latent to style
        style = self.style_net(latent)

        # Concatenate obs and style
        h = torch.cat([obs, style], dim=-1)

        # Process through MLP
        for layer in self.mlp:
            h = layer(h)

        return h

    def get_output_dim(self) -> int:
        """Get output dimension.

        Returns:
            Output dimension.
        """
        # Find last linear layer
        for layer in reversed(self.mlp):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        return self.output_dim


class LatentConditionedMLP(nn.Module):
    """Simple latent-conditioned MLP (for critic).

    Concatenates observations with latents and processes through MLP.
    Based on AMPMLPNet from old code.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list = [1024, 1024, 512],
        activation: str = "relu",
    ):
        """Initialize latent-conditioned MLP.

        Args:
            obs_dim: Dimension of observations.
            latent_dim: Dimension of latent codes.
            output_dim: Dimension of output.
            hidden_dims: List of hidden layer dimensions.
            activation: Activation function name.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Build MLP
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "elu":
            act_fn = nn.ELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()

        layers = []
        in_dim = obs_dim + latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Initialize
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Observations [batch, obs_dim].
            latent: Latent codes [batch, latent_dim].

        Returns:
            Output features [batch, output_dim].
        """
        # Concatenate and process
        h = torch.cat([obs, latent], dim=-1)
        h = self.mlp(h)
        return h

    def get_output_dim(self) -> int:
        """Get output dimension.

        Returns:
            Output dimension.
        """
        # Find last linear layer
        for layer in reversed(self.mlp):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        return self.output_dim


class AWDActorCritic(nn.Module):
    """AWD Actor-Critic network with discriminator and encoder.

    This combines:
    - Style-conditioned actor network
    - Latent-conditioned critic network
    - Discriminator network
    - Encoder network
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        amp_obs_dim: int,
        latent_dim: int = 64,
        actor_hidden_dims: list = [1024, 1024, 512],
        critic_hidden_dims: list = [1024, 1024, 512],
        disc_hidden_dims: list = [1024, 512],
        enc_hidden_dims: list = [1024, 512],
        style_hidden_dims: list = [512, 256],
        style_dim: int = 64,
        activation: str = "relu",
        init_noise_std: float = 1.0,
        separate_encoder: bool = False,
    ):
        """Initialize AWD actor-critic.

        Args:
            obs_dim: Dimension of observations.
            action_dim: Dimension of actions.
            amp_obs_dim: Dimension of AMP observations.
            latent_dim: Dimension of latent codes.
            actor_hidden_dims: Hidden dims for actor.
            critic_hidden_dims: Hidden dims for critic.
            disc_hidden_dims: Hidden dims for discriminator.
            enc_hidden_dims: Hidden dims for encoder.
            style_hidden_dims: Hidden dims for style processing.
            style_dim: Dimension of style vectors.
            activation: Activation function name.
            init_noise_std: Initial noise std for action distribution.
            separate_encoder: If True, encoder has separate network. If False, shares with disc.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.amp_obs_dim = amp_obs_dim
        self.latent_dim = latent_dim

        # Actor network (style-conditioned)
        self.actor = StyleConditionedMLP(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            output_dim=actor_hidden_dims[-1],
            hidden_dims=actor_hidden_dims,
            style_hidden_dims=style_hidden_dims,
            style_dim=style_dim,
            activation=activation,
        )

        actor_out_dim = self.actor.get_output_dim()

        # Actor head (mean)
        self.actor_mean = nn.Linear(actor_out_dim, action_dim)
        nn.init.kaiming_uniform_(self.actor_mean.weight, nonlinearity="relu")
        nn.init.zeros_(self.actor_mean.bias)

        # Actor head (log std)
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * torch.log(torch.tensor(init_noise_std))
        )

        # Critic network (latent-conditioned)
        self.critic = LatentConditionedMLP(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            output_dim=critic_hidden_dims[-1],
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        critic_out_dim = self.critic.get_output_dim()

        # Critic head
        self.critic_head = nn.Linear(critic_out_dim, 1)
        nn.init.kaiming_uniform_(self.critic_head.weight, nonlinearity="relu")
        nn.init.zeros_(self.critic_head.bias)

        # Discriminator
        self.discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim, hidden_dims=disc_hidden_dims, activation=activation
        )

        # Encoder
        self.encoder = AWDEncoder(
            input_dim=amp_obs_dim,
            latent_dim=latent_dim,
            hidden_dims=enc_hidden_dims,
            activation=activation,
            separate_network=separate_encoder,
            disc_network=self.discriminator if not separate_encoder else None,
        )

    def act(
        self, obs: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action distribution.

        Args:
            obs: Observations [batch, obs_dim].
            latents: Latent codes [batch, latent_dim].

        Returns:
            Tuple of (actions, log_probs, mu, std).
        """
        # Get actor output
        actor_features = self.actor(obs, latents)
        mu = self.actor_mean(actor_features)
        mu = torch.tanh(mu)  # Bound actions

        # Sample actions
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mu, std)
        actions = dist.sample()
        # RSL-RL expects log_probs per action dimension, not summed
        log_probs = dist.log_prob(actions)

        return actions, log_probs, mu, std

    def act_inference(
        self, obs: torch.Tensor, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions for inference (same as act but with gradient tracking).

        Args:
            obs: Observations [batch, obs_dim].
            latents: Latent codes [batch, latent_dim].

        Returns:
            Tuple of (actions, log_probs, mu, std).
        """
        return self.act(obs, latents)

    def evaluate(
        self,
        critic_obs: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions and compute values.

        Args:
            critic_obs: Critic observations [batch, obs_dim].
            obs: Actor observations [batch, obs_dim].
            actions: Actions [batch, action_dim].
            latents: Latent codes [batch, latent_dim].

        Returns:
            Tuple of (action_log_probs, entropy, values, mu, std).
        """
        # Actor evaluation
        actor_features = self.actor(obs, latents)
        mu = self.actor_mean(actor_features)
        mu = torch.tanh(mu)

        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mu, std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1)

        # Critic evaluation
        critic_features = self.critic(critic_obs, latents)
        values = self.critic_head(critic_features)

        return action_log_probs, entropy, values, mu, std

    def get_disc_logit_weights(self) -> torch.Tensor:
        """Get discriminator logit weights.

        Returns:
            Flattened logit weights.
        """
        return self.discriminator.get_logit_weights()

    def get_disc_weights(self) -> list:
        """Get all discriminator weights.

        Returns:
            List of flattened weight tensors.
        """
        return self.discriminator.get_all_weights()

    def get_enc_weights(self) -> list:
        """Get all encoder weights.

        Returns:
            List of flattened weight tensors.
        """
        return self.encoder.get_weights()
