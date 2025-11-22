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

"""Motion visualization environment - plays back reference motions.

Useful for debugging and validating motion data.
Migrated from IsaacGym's DucklingViewMotion to IsaacLab.
"""

import torch
import numpy as np
from typing import Tuple

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
class DucklingViewMotionCfg(DucklingAMPCfg):
    """Configuration for motion viewing environment.

    Extends DucklingAMPCfg with motion playback parameters.
    """

    # Disable PD control for pure kinematic playback
    pd_control: bool = False
    """Whether to use PD control (False for kinematic playback)"""


##
# Environment
##

class DucklingViewMotion(DucklingAMP):
    """Motion visualization environment.

    Plays back reference motion data kinematically (no physics control).
    Useful for:
    - Validating motion data
    - Debugging motion library
    - Visualizing reference motions
    """

    cfg: DucklingViewMotionCfg

    def __init__(self, cfg: DucklingViewMotionCfg, render_mode: str | None = None, **kwargs):
        """Initialize motion viewing environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
        """
        # Override num_observations BEFORE parent init
        # ViewMotion has NO task_obs (only base duckling_obs)
        # Total: 51 (base) + 0 (task_obs) = 51
        base_obs_size = 51
        task_obs_size = 0  # ViewMotion has no task observations
        cfg.num_observations = base_obs_size + task_obs_size

        # Store motion dt
        self._motion_dt = cfg.sim.dt * cfg.decimation

        # Disable PD control and use kinematic mode
        cfg.pd_control = False

        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)

        # Assign motions to environments
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)

        # Tracking for debugging
        self.accumulated_key_pos_anim = []
        self.accumulated_key_pos_sim = []

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.

        For motion viewing, we disable actuation forces.

        Args:
            actions: Actions (ignored for motion viewing).
        """
        # Store actions but don't apply forces
        self.actions[:] = actions

        # Zero out all joint forces (kinematic mode)
        # In IsaacLab, we don't apply any control
        # The robot will be set kinematically in _post_physics_step

    def _post_physics_step(self):
        """Process environment after physics step."""
        # Call parent
        super()._post_physics_step()

        # Synchronize robot with motion data
        self._motion_sync()

    def _motion_sync(self):
        """Synchronize robot state with reference motion."""
        # Get current motion time
        motion_ids = self._motion_ids
        motion_times = self.episode_length_buf.float() * self._motion_dt

        # Get motion state
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Track key positions for debugging
        if hasattr(self, '_robot'):
            key_body_pos = self._robot.data.body_pos_w[:, self.cfg.key_body_ids, :]
            self.accumulated_key_pos_anim.append(key_pos.detach().cpu().numpy())
            self.accumulated_key_pos_sim.append(key_body_pos.detach().cpu().numpy())

        # Set environment state kinematically
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Set root pose
        self._robot.write_root_pose_to_sim(root_pos, root_rot, env_ids)

        # Set root velocity
        root_velocity = torch.cat([root_vel, root_ang_vel], dim=-1)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)

        # Set joint state
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation signals.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        # Get motion lengths
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)

        # Compute reset
        terminated, truncated = compute_view_motion_reset(
            motion_lengths,
            self.episode_length_buf,
            self._motion_dt,
        )

        # Save accumulated data on reset
        if terminated.any():
            accumulated_key_pos_anim = np.array(self.accumulated_key_pos_anim)
            accumulated_key_pos_sim = np.array(self.accumulated_key_pos_sim)

            np.save("anim", accumulated_key_pos_anim)
            np.save("sim", accumulated_key_pos_sim)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Cycle to next motion
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(
            self._motion_ids[env_ids] + self.num_envs, num_motions
        )

        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Don't reset actor state - will be set by motion sync


##
# JIT Compiled Helper Functions
##

@torch.jit.script
def compute_view_motion_reset(
    motion_lengths: torch.Tensor,
    progress_buf: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute reset conditions for motion viewing.

    Args:
        motion_lengths: Length of each motion [num_envs]
        progress_buf: Progress buffer [num_envs]
        dt: Time step

    Returns:
        Tuple of (terminated, truncated) boolean tensors
    """
    # Reset when motion is complete
    motion_times = progress_buf.float() * dt
    terminated = motion_times > motion_lengths

    # No truncation (let motion play to completion)
    truncated = torch.zeros_like(terminated)

    return terminated, truncated
