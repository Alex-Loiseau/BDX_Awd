"""Duckling Command environment with AMP observations - IsaacLab version.

This extends DucklingCommandEnv to add AMP observation computation for AWD training.
"""

import torch
from typing import Dict

from .duckling_command_env import DucklingCommandEnv, DucklingCommandCfg
from .amp_observations import AMPObservationMixin
from isaaclab.utils.configclass import configclass


@configclass
class DucklingCommandAMPCfg(DucklingCommandCfg):
    """Configuration for DucklingCommand with AMP."""

    # AMP-specific settings
    num_amp_obs_steps: int = 2  # Number of timesteps in AMP observation history

    # Motion file for demonstrations (optional, can be set later)
    motion_file: str = ""


class DucklingCommandAMPEnv(AMPObservationMixin, DucklingCommandEnv):
    """DucklingCommand environment with AMP observation support.

    This adds:
    - AMP observation computation
    - Demonstration loading
    - Support for AWD/AMP training
    """

    cfg: DucklingCommandAMPCfg

    def __init__(self, cfg: DucklingCommandAMPCfg, render_mode: str | None = None, **kwargs):
        """Initialize DucklingCommand with AMP.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for visualization.
        """
        # Initialize both parent classes
        # AMPObservationMixin first to set up AMP-specific attributes
        # Then DucklingCommandEnv for environment setup
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize AMP observation buffer (after robot is created)
        if hasattr(self, '_robot') and self._robot is not None:
            self._init_amp_obs_buf()

    def _setup_scene(self):
        """Setup scene including robot and AMP initialization.

        Overrides parent to add AMP buffer initialization.
        """
        # Call parent to setup robot and scene
        super()._setup_scene()

        # Now robot exists, initialize AMP buffers
        self._init_amp_obs_buf()

        # Note: We don't compute initial AMP observations here because
        # robot data is not yet available. The buffer will be filled
        # after the first physics step.

    def _apply_action(self) -> None:
        """Apply actions and update AMP observations.

        Extends parent to update AMP observation history.
        """
        # Update historical AMP observations before step
        self._update_hist_amp_obs()

        # Apply action (parent implementation)
        super()._apply_action()

        # Compute current AMP observations after step
        # This will be called after physics step in post_physics_step

    def step(self, action: torch.Tensor):
        """Step environment and compute AMP observations.

        Args:
            action: Action tensor.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # Store action
        self.actions = action

        # Execute parent step
        obs_dict, rew, terminated, truncated, extras = super().step(action)

        # Compute AMP observations for current timestep
        self._compute_amp_observations()

        # Add AMP observations to extras/info
        amp_obs = self.get_amp_observations()
        extras["amp_obs"] = amp_obs

        return obs_dict, rew, terminated, truncated, extras

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset environments and their AMP observation history.

        Args:
            env_ids: Environment indices to reset.
        """
        # Reset parent
        super()._reset_idx(env_ids)

        # Reset AMP observation buffers for these environments
        if hasattr(self, '_amp_obs_buf') and self._amp_obs_buf is not None:
            self._amp_obs_buf[env_ids] = 0.0

            # Recompute AMP observations for reset envs
            self._compute_amp_observations(env_ids)
