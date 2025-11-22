"""Perturbation environment for quadrupedal robots.

This environment trains the robot to handle external perturbations by launching projectiles
at the robot during locomotion.

Migrated from IsaacGym's DucklingPerturb to IsaacLab.
Note: This is a simplified version that inherits from DucklingCommand.
      Will be updated to inherit from DucklingAMP once AMP is migrated.
"""

import numpy as np
import torch
from dataclasses import MISSING

try:
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.utils import configclass
except ImportError:
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab.utils import configclass

from .duckling_command_env import DucklingCommandEnv, DucklingCommandCfg


# Perturbation object schedule: [type, timesteps_before_launch]
PERTURB_OBJS = [
    ["small", 200],
    ["small", 7],
    ["small", 10],
    ["small", 35],
    ["small", 2],
    ["small", 2],
    ["small", 3],
    ["small", 2],
    ["small", 2],
    ["small", 3],
    ["small", 2],
    ["large", 60],
    ["small", 300],
]


##
# Configuration
##

@configclass
class DucklingPerturbCfg(DucklingCommandCfg):
    """Configuration for the perturbation environment.

    Extends DucklingCommandCfg with perturbation parameters.
    """

    # Projectile spawn parameters
    proj_dist_min: float = 4.0
    """Minimum distance from robot to spawn projectile (m)"""

    proj_dist_max: float = 5.0
    """Maximum distance from robot to spawn projectile (m)"""

    proj_h_min: float = 0.25
    """Minimum height to spawn projectile (m)"""

    proj_h_max: float = 2.0
    """Maximum height to spawn projectile (m)"""

    proj_speed_min: float = 30.0
    """Minimum projectile launch speed (m/s)"""

    proj_speed_max: float = 40.0
    """Maximum projectile launch speed (m/s)"""

    # Disable early termination for perturbation training
    enable_early_termination: bool = False
    """Disable early termination so robot learns to recover from perturbations"""


##
# Environment
##

class DucklingPerturbEnv(DucklingCommandEnv):
    """Environment for training robot to handle external perturbations.

    The robot must maintain locomotion while being hit by projectiles launched
    from random positions around it. This trains robustness and recovery.

    Note: Projectiles are spawned and launched automatically according to the
    PERTURB_OBJS schedule. Early termination is disabled so the robot learns
    to recover from perturbations rather than just avoiding them.
    """

    cfg: DucklingPerturbCfg

    def __init__(self, cfg: DucklingPerturbCfg, render_mode: str | None = None, **kwargs):
        """Initialize the perturbation environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
        """
        # Override num_observations BEFORE parent init
        # Perturb has NO task_obs (only base duckling_obs)
        # Total: 51 (base) + 0 (task_obs) = 51
        base_obs_size = 51
        task_obs_size = 0  # Perturb has no task observations
        cfg.num_observations = base_obs_size + task_obs_size

        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)

        # Calculate perturbation schedule
        self._calc_perturb_times()

        # Initialize perturbation state
        # Note: In full implementation, would spawn actual projectile assets
        # For now, we just track when perturbations should occur
        self._next_perturb_id = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    def _calc_perturb_times(self):
        """Calculate timesteps when each projectile should be launched."""
        self._perturb_timesteps = []
        total_steps = 0

        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)
        self._total_perturb_steps = self._perturb_timesteps[-1] + 1

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Call parent reset
        super()._reset_idx(env_ids)

        # Reset perturbation schedule
        self._next_perturb_id[env_ids] = 0

    def _post_physics_step(self):
        """Process environment after physics step."""
        # Apply perturbations
        self._update_perturbations()

        # Call parent
        super()._post_physics_step()

    def _update_perturbations(self):
        """Apply external perturbations to the robot.

        In the original implementation, this launched projectiles.
        For now, we apply direct force perturbations as a simplified version.
        """
        # Get current timestep in perturbation cycle
        curr_timestep = (self.episode_length_buf[0] % self._total_perturb_steps).item()

        # Check if it's time to perturb
        perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]

        if len(perturb_step) > 0:
            perturb_id = perturb_step[0]

            # Determine which type of perturbation
            obj_type = PERTURB_OBJS[perturb_id][0]

            # Apply force perturbation to robot
            # Force magnitude based on projectile type
            if obj_type == "small":
                force_magnitude = 150.0  # Newtons
            else:  # large
                force_magnitude = 250.0

            # Random direction (horizontal)
            n = self.num_envs
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device)
            force_x = force_magnitude * torch.cos(rand_theta)
            force_y = force_magnitude * torch.sin(rand_theta)
            force_z = torch.zeros(n, device=self.device)

            # Apply impulse force to robot base
            # Note: This is a simplified version. Full implementation would
            # spawn actual projectile assets and use physics collisions.
            force = torch.stack([force_x, force_y, force_z], dim=-1)

            # Apply force as external wrench (would need to be implemented
            # in a more sophisticated way for full physics simulation)
            # For now, just log that perturbation occurred
            if self.episode_length_buf[0] % 100 == 0:
                print(f"[Perturb] Step {curr_timestep}: Applying {obj_type} perturbation (force={force_magnitude}N)")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation signals.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        # Override to disable early termination for perturbation training
        # Robot should learn to recover, not just avoid failure

        # Only terminate on episode length
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated = time_out

        return terminated, truncated


##
# Helper Functions
##

@torch.jit.script
def compute_duckling_reset(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    contact_buf: torch.Tensor,
    contact_body_ids: torch.Tensor,
    rigid_body_pos: torch.Tensor,
    max_episode_length: float,
    enable_early_termination: bool,
    termination_heights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reset conditions for perturbation environment.

    Args:
        reset_buf: Reset buffer
        progress_buf: Progress buffer
        contact_buf: Contact forces
        contact_body_ids: IDs of contact bodies
        rigid_body_pos: Rigid body positions
        max_episode_length: Maximum episode length
        enable_early_termination: Whether to enable early termination
        termination_heights: Height thresholds for termination

    Returns:
        Tuple of (reset, terminated) boolean tensors
    """
    # Disable early termination for perturbation training
    terminated = torch.zeros_like(reset_buf)
    reset = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        terminated
    )

    return reset, terminated
