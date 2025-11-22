# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""AMP Task environment - combines AMP with task-specific goals.

Migrated from IsaacGym's DucklingAMPTask to IsaacLab.
"""

import torch
import pickle

try:
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.utils import configclass
except ImportError:
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab.utils import configclass

from .duckling_amp import DucklingAMP, DucklingAMPCfg


##
# Configuration
##

@configclass
class DucklingAMPTaskCfg(DucklingAMPCfg):
    """Configuration for AMP Task environment.

    Extends DucklingAMPCfg with task-specific parameters.
    """

    enable_task_obs: bool = True
    """Whether to include task-specific observations"""

    debug_save_obs: bool = False
    """Whether to save observations for debugging"""


##
# Environment
##

class DucklingAMPTask(DucklingAMP):
    """AMP environment with task-specific goals.

    Base class for AMP environments that combine natural motion with
    task objectives (e.g., reaching a target location).
    """

    cfg: DucklingAMPTaskCfg

    def __init__(self, cfg: DucklingAMPTaskCfg, render_mode: str | None = None, **kwargs):
        """Initialize AMP Task environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
        """
        # Override num_observations BEFORE parent init
        # AMPTask has task_obs_size = 0 (base class)
        # Subclasses (Command, Heading) will override this in their own __init__
        # Total: 51 (base) + 0 (task_obs) = 51
        base_obs_size = 51
        task_obs_size = 0  # AMPTask base has no task observations
        cfg.num_observations = base_obs_size + task_obs_size

        self._enable_task_obs = cfg.enable_task_obs

        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)

        # Debug observation saving
        if self.cfg.debug_save_obs:
            self.saved_obs = []

    def get_task_obs_size(self) -> int:
        """Get size of task-specific observations.

        Returns:
            Task observation size (0 for base class).
        """
        return 0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.

        Args:
            actions: Actions from the policy.
        """
        # Call parent
        super()._pre_physics_step(actions)

        # Update task
        self._update_task()

    def _get_observations(self) -> dict:
        """Compute observations.

        Returns:
            Dictionary with 'policy' observations.
        """
        # Get base AMP observations
        duckling_obs = super()._get_observations()["policy"]

        # Add task observations if enabled
        if self._enable_task_obs:
            task_obs = self._compute_task_obs()
            obs = torch.cat([duckling_obs, task_obs], dim=-1)
        else:
            obs = duckling_obs

        # Debug: save observations
        if self.cfg.debug_save_obs:
            self.saved_obs.append(obs[0].cpu().numpy())
            pickle.dump(self.saved_obs, open("saved_obs.pkl", "wb"))

        return {"policy": obs}

    def _compute_task_obs(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Compute task-specific observations.

        Args:
            env_ids: Environment IDs to compute for (None = all).

        Returns:
            Task observations.
        """
        # Base implementation returns empty
        # Subclasses should override this
        if env_ids is None:
            return torch.zeros((self.num_envs, self.get_task_obs_size()), device=self.device)
        else:
            return torch.zeros((len(env_ids), self.get_task_obs_size()), device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards.

        Returns:
            Rewards for each environment.
        """
        # Base implementation - should be overridden by subclasses
        # For now, return zero rewards
        return torch.zeros(self.num_envs, device=self.device)

    def _update_task(self):
        """Update task state.

        Called before each physics step. Subclasses can override
        to update task-specific state (e.g., target positions).
        """
        pass

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Call parent reset
        super()._reset_idx(env_ids)

        # Reset task
        self._reset_task(env_ids)

    def _reset_task(self, env_ids: torch.Tensor):
        """Reset task for specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Base implementation does nothing
        # Subclasses can override to reset task state
        pass

    def _draw_task(self):
        """Draw task visualization.

        Called during rendering. Subclasses can override to draw
        task-specific visualizations (e.g., target markers).
        """
        pass
