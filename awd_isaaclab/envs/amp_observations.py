"""AMP Observation utilities for IsaacLab environments.

This module provides functions to compute AMP observations for motion imitation.
Ported from old_awd/env/tasks/duckling_amp.py
"""

import torch
from typing import List
from isaaclab.utils.math import quat_mul, quat_rotate_inverse


def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Compute inverse of heading quaternion (rotation around z-axis only).

    Args:
        q: Quaternion (w, x, y, z) [batch, 4].

    Returns:
        Inverse heading quaternion [batch, 4].
    """
    # Extract heading (yaw) from quaternion
    # For a unit quaternion, yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    # We can reconstruct heading quaternion as (cos(yaw/2), 0, 0, sin(yaw/2))

    # Convert to heading quaternion (only yaw component)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # Compute yaw angle
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # Create heading quaternion
    heading_q = torch.zeros_like(q)
    heading_q[:, 0] = torch.cos(yaw / 2)  # w
    heading_q[:, 1] = 0  # x
    heading_q[:, 2] = 0  # y
    heading_q[:, 3] = torch.sin(yaw / 2)  # z

    # Return inverse (conjugate for unit quaternions)
    heading_q_inv = heading_q.clone()
    heading_q_inv[:, 1:] *= -1

    return heading_q_inv


def build_amp_observations(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    root_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
    local_root_obs: bool,
    root_height_obs: bool,
    dof_obs_size: int,
    dof_offsets: List[int],
) -> torch.Tensor:
    """Build AMP observations from robot state.

    Args:
        root_pos: Root position [batch, 3].
        root_rot: Root rotation quaternion (w, x, y, z) [batch, 4].
        root_vel: Root linear velocity [batch, 3].
        root_ang_vel: Root angular velocity [batch, 3].
        dof_pos: DOF positions [batch, num_dof].
        dof_vel: DOF velocities [batch, num_dof].
        key_body_pos: Key body positions [batch, num_key_bodies, 3].
        local_root_obs: Whether to use local root observations.
        root_height_obs: Whether to include root height.
        dof_obs_size: Size of DOF observations.
        dof_offsets: DOF offsets for different body parts.

    Returns:
        AMP observations [batch, num_amp_obs].
    """
    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(root_rot)

    obs = []

    if local_root_obs:
        # Root orientation relative to heading
        root_rot_obs = quat_mul(heading_rot_inv, root_rot)
    else:
        root_rot_obs = root_rot

    # Root rotation (4D quaternion)
    obs.append(root_rot_obs)

    if root_height_obs:
        # Root height
        obs.append(root_h)

    # Local root linear velocity
    local_root_vel = quat_rotate_inverse(heading_rot_inv, root_vel)
    obs.append(local_root_vel)

    # Local root angular velocity
    local_root_ang_vel = quat_rotate_inverse(heading_rot_inv, root_ang_vel)
    obs.append(local_root_ang_vel)

    # DOF positions
    obs.append(dof_pos[:, :dof_obs_size])

    # DOF velocities
    obs.append(dof_vel[:, :dof_obs_size])

    # Key body positions (local to root)
    if key_body_pos.shape[1] > 0:
        num_key_bodies = key_body_pos.shape[1]
        flat_key_pos = key_body_pos.reshape(key_body_pos.shape[0], -1)  # [batch, num_key_bodies * 3]

        # Transform to local frame
        local_key_body_pos = key_body_pos - root_pos.unsqueeze(1)  # [batch, num_key_bodies, 3]
        local_key_body_pos = quat_rotate_inverse(
            heading_rot_inv.unsqueeze(1).expand(-1, num_key_bodies, -1).reshape(-1, 4),
            local_key_body_pos.reshape(-1, 3),
        ).reshape(key_body_pos.shape[0], -1)  # [batch, num_key_bodies * 3]

        obs.append(local_key_body_pos)

    # Concatenate all observations
    obs = torch.cat(obs, dim=-1)

    return obs


class AMPObservationMixin:
    """Mixin class to add AMP observation computation to environments.

    This should be mixed into DirectRLEnv-based environments.
    """

    def __init__(self, *args, **kwargs):
        """Initialize AMP observation mixin."""
        # Set AMP configuration BEFORE calling super().__init__
        # This ensures attributes are available when _setup_scene is called
        cfg = args[0] if args else kwargs.get('cfg')
        self._amp_cfg = cfg  # Store cfg reference for later use
        self.num_amp_obs_steps = getattr(cfg, "num_amp_obs_steps", 2)
        self._local_root_obs = getattr(cfg, "local_root_obs", True)
        self._root_height_obs = getattr(cfg, "root_height_obs", True)

        # Will be initialized after robot is created
        self._num_amp_obs_per_step = None
        self._amp_obs_buf = None
        self._key_body_ids = None

        # Call parent init
        super().__init__(*args, **kwargs)

    def _init_amp_obs_buf(self):
        """Initialize AMP observation buffer.

        Should be called after robot and scene are created.
        """
        # Compute AMP observation size
        self._num_amp_obs_per_step = self._compute_amp_obs_size()

        # Total AMP observations across all timesteps
        self.num_amp_obs = self.num_amp_obs_steps * self._num_amp_obs_per_step

        # Create buffer for AMP observations
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self.num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )

        # Current and historical observations
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

    def _compute_amp_obs_size(self) -> int:
        """Compute size of AMP observations per timestep.

        Returns:
            Number of AMP observation elements per timestep.
        """
        # Root orientation (quaternion): 4
        obs_size = 4

        # Root height: 1 (if enabled)
        if self._root_height_obs:
            obs_size += 1

        # Root linear velocity: 3
        obs_size += 3

        # Root angular velocity: 3
        obs_size += 3

        # DOF positions: num_dof
        # Use cfg.num_actions since self.num_actions may not be set yet during init
        num_dof = self._amp_cfg.num_actions if hasattr(self, '_amp_cfg') else self.num_actions
        obs_size += num_dof

        # DOF velocities: num_dof
        obs_size += num_dof

        # Key body positions: num_key_bodies * 3
        if hasattr(self, "_key_body_ids") and self._key_body_ids is not None:
            obs_size += len(self._key_body_ids) * 3

        return obs_size

    def _compute_amp_observations(self, env_ids=None):
        """Compute AMP observations for current timestep.

        Args:
            env_ids: Environment IDs to compute for (None = all).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Get robot state
        root_pos = self._robot.data.root_pos_w[env_ids]
        root_rot = self._robot.data.root_quat_w[env_ids]  # (w, x, y, z)
        root_vel = self._robot.data.root_lin_vel_w[env_ids]
        root_ang_vel = self._robot.data.root_ang_vel_w[env_ids]
        dof_pos = self._robot.data.joint_pos[env_ids]
        dof_vel = self._robot.data.joint_vel[env_ids]

        # Get key body positions (if available)
        if hasattr(self, "_key_body_ids") and self._key_body_ids is not None:
            key_body_pos = self._robot.data.body_pos_w[env_ids][:, self._key_body_ids]
        else:
            key_body_pos = torch.zeros((len(env_ids), 0, 3), device=self.device)

        # Build AMP observations
        # Use cfg.num_actions if self.num_actions not available yet
        num_dof = self._amp_cfg.num_actions if hasattr(self, '_amp_cfg') else self.num_actions
        amp_obs = build_amp_observations(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            key_body_pos=key_body_pos,
            local_root_obs=self._local_root_obs,
            root_height_obs=self._root_height_obs,
            dof_obs_size=num_dof,
            dof_offsets=[],  # Not used in current implementation
        )

        # Store in buffer
        self._curr_amp_obs_buf[env_ids] = amp_obs

    def _update_hist_amp_obs(self, env_ids=None):
        """Update historical AMP observations.

        Args:
            env_ids: Environment IDs to update (None = all).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Shift history: [t-1, t-2, ...] -> [t-2, t-3, ...]
        for i in range(self.num_amp_obs_steps - 1, 1, -1):
            self._amp_obs_buf[env_ids, i] = self._amp_obs_buf[env_ids, i - 1].clone()

        # Move current to history: [t-1] <- [t]
        if self.num_amp_obs_steps > 1:
            self._amp_obs_buf[env_ids, 1] = self._curr_amp_obs_buf[env_ids].clone()

    def get_amp_observations(self) -> torch.Tensor:
        """Get flattened AMP observations.

        Returns:
            AMP observations [num_envs, num_amp_obs].
        """
        amp_obs_flat = self._amp_obs_buf.view(self.num_envs, -1)
        return amp_obs_flat

    def fetch_amp_obs_demo(self, num_samples: int) -> torch.Tensor:
        """Fetch demonstration AMP observations.

        This is a placeholder - actual implementation should load from motion files.

        Args:
            num_samples: Number of demonstration samples.

        Returns:
            Demo AMP observations [num_samples, num_amp_obs].
        """
        # For now, return zeros
        # TODO: Implement motion library loading
        return torch.zeros((num_samples, self.num_amp_obs), device=self.device)
