"""Heading-following environment for quadrupedal robots.

This environment trains the robot to move in a target direction at a target speed
while maintaining a target heading (facing direction).

Migrated from IsaacGym's DucklingHeading to IsaacLab.
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


##
# Configuration
##

@configclass
class DucklingHeadingCfg(DucklingCommandCfg):
    """Configuration for the heading-following environment.

    Extends DucklingCommandCfg with heading control parameters.
    """

    # Heading-specific parameters
    tar_speed_min: float = 0.0
    """Minimum target speed (m/s)"""

    tar_speed_max: float = 1.0
    """Maximum target speed (m/s)"""

    heading_change_steps_min: int = 100
    """Minimum steps before heading changes"""

    heading_change_steps_max: int = 300
    """Maximum steps before heading changes"""

    enable_rand_heading: bool = True
    """Whether to randomize heading direction"""


##
# Environment
##

class DucklingHeadingEnv(DucklingCommandEnv):
    """Environment for training robot to follow heading and speed commands.

    The robot must:
    1. Move at a target speed
    2. Move in a target direction
    3. Face a target heading (which may be different from movement direction)

    This allows training side-stepping and other complex maneuvers.
    """

    cfg: DucklingHeadingCfg

    def __init__(self, cfg: DucklingHeadingCfg, render_mode: str | None = None, **kwargs):
        """Initialize the heading environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
        """
        # Override num_observations BEFORE parent init
        # Heading has task_obs = 5 (local_tar_dir=2 + tar_speed=1 + local_tar_face_dir=2)
        # Total: 51 (base) + 5 (task_obs) = 56
        base_obs_size = 51
        task_obs_size = 5 if cfg.enable_task_obs else 0
        cfg.num_observations = base_obs_size + task_obs_size

        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)

        # Heading-specific state
        self._heading_change_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self._tar_speed = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._tar_dir = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float32
        )
        self._tar_dir[:, 0] = 1.0  # Default: move forward

        self._tar_facing_dir = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float32
        )
        self._tar_facing_dir[:, 0] = 1.0  # Default: face forward

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.

        Args:
            actions: Actions from the policy.
        """
        # Store previous position for velocity calculation
        self._prev_root_pos[:] = self._robot.data.root_state_w[:, :3]

        # Call parent
        super()._pre_physics_step(actions)

    def _get_observations(self) -> dict:
        """Compute observations.

        Returns:
            Dictionary with 'policy' observations.
        """
        # Get base observations from parent
        obs_dict = super()._get_observations()

        # Add heading-specific task observations
        if self.cfg.enable_task_obs:
            task_obs = self._compute_task_obs()
            # Concatenate with base observations
            obs_dict["policy"] = torch.cat([obs_dict["policy"], task_obs], dim=-1)

        return obs_dict

    def _compute_task_obs(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Compute heading task observations.

        Args:
            env_ids: Environment IDs to compute observations for. If None, compute for all.

        Returns:
            Tensor of shape (num_envs, 5) with heading observations.
        """
        if env_ids is None:
            root_state = self._robot.data.root_state_w
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
        else:
            root_state = self._robot.data.root_state_w[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]

        # Compute heading observations
        obs = compute_heading_observations(root_state, tar_dir, tar_speed, tar_face_dir)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for heading task.

        Returns:
            Tensor of shape (num_envs,) with rewards.
        """
        # Compute heading-specific rewards
        root_pos = self._robot.data.root_state_w[:, :3]
        root_rot = self._robot.data.root_state_w[:, 3:7]

        reward = compute_heading_reward(
            root_pos,
            self._prev_root_pos,
            root_rot,
            self._tar_dir,
            self._tar_speed,
            self._tar_facing_dir,
            self.step_dt
        )

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation signals.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        # Get base termination conditions
        terminated, truncated = super()._get_dones()

        # No additional termination for heading task
        # (task resets are handled in _update_task)

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Call parent reset
        super()._reset_idx(env_ids)

        # Reset heading-specific state
        self._prev_root_pos[env_ids] = self._robot.data.root_state_w[env_ids, :3]

        # Reset heading task
        self._reset_task(env_ids)

    def _update_task(self):
        """Update heading task (change target heading periodically)."""
        # Check if it's time to change heading for any environment
        reset_task_mask = self.episode_length_buf >= self._heading_change_steps
        reset_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        if len(reset_env_ids) > 0:
            self._reset_task(reset_env_ids)

    def _reset_task(self, env_ids: torch.Tensor):
        """Reset heading task for specific environments.

        Args:
            env_ids: Environment IDs to reset task for.
        """
        n = len(env_ids)

        # Generate random heading if enabled
        if self.cfg.enable_rand_heading:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        else:
            rand_theta = torch.zeros(n, device=self.device)
            rand_face_theta = torch.zeros(n, device=self.device)

        # Compute target direction from angle
        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)

        # Random target speed
        tar_speed = (self.cfg.tar_speed_max - self.cfg.tar_speed_min) * torch.rand(
            n, device=self.device
        ) + self.cfg.tar_speed_min

        # Random time before next heading change
        change_steps = torch.randint(
            low=self.cfg.heading_change_steps_min,
            high=self.cfg.heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        # Compute target facing direction
        face_tar_dir = torch.stack(
            [torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1
        )

        # Update task state
        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.episode_length_buf[env_ids] + change_steps

    def _post_physics_step(self):
        """Process environment after physics step."""
        # Call parent
        super()._post_physics_step()

        # Update heading task
        self._update_task()


##
# JIT Compiled Helper Functions
# Note: Must define utility functions first, then use them in main functions
##

@torch.jit.script
def calc_heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract heading (yaw-only) quaternion from full quaternion.

    Args:
        quat: Quaternion (w, x, y, z) [num_envs, 4]

    Returns:
        Heading quaternion [num_envs, 4]
    """
    # Extract yaw angle from quaternion
    # For quat = (w, x, y, z), yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # Create heading quaternion (rotation around Z axis only)
    heading_quat = torch.zeros_like(quat)
    heading_quat[:, 0] = torch.cos(yaw / 2)  # w
    heading_quat[:, 3] = torch.sin(yaw / 2)  # z

    return heading_quat


@torch.jit.script
def calc_heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
    """Compute inverse of heading quaternion.

    Args:
        quat: Quaternion (w, x, y, z) [num_envs, 4]

    Returns:
        Inverse heading quaternion [num_envs, 4]
    """
    heading_quat = calc_heading_quat(quat)
    # Inverse of unit quaternion is conjugate
    inv_quat = heading_quat.clone()
    inv_quat[:, 1:] = -inv_quat[:, 1:]  # Negate x, y, z
    return inv_quat


@torch.jit.script
def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion.

    Args:
        quat: Quaternion (w, x, y, z) [num_envs, 4]
        vec: Vector [num_envs, 3]

    Returns:
        Rotated vector [num_envs, 3]
    """
    # Extract quaternion components
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]

    # Quaternion rotation formula: v' = q * v * q^-1
    # Optimized version without explicitly computing q * v * q^-1

    # Intermediate terms
    t0 = 2.0 * (w * vx + y * vz - z * vy)
    t1 = 2.0 * (w * vy + z * vx - x * vz)
    t2 = 2.0 * (w * vz + x * vy - y * vx)

    # Rotated vector
    rx = vx + x * t0 + y * t1 + z * t2
    ry = vy - y * t0 + w * t1 + x * t2
    rz = vz - z * t0 - x * t1 + w * t2

    return torch.stack([rx, ry, rz], dim=-1)


@torch.jit.script
def compute_heading_observations(
    root_state: torch.Tensor,
    tar_dir: torch.Tensor,
    tar_speed: torch.Tensor,
    tar_face_dir: torch.Tensor
) -> torch.Tensor:
    """Compute heading observations in robot's local frame.

    Args:
        root_state: Root state (pos, quat, lin_vel, ang_vel) [num_envs, 13]
        tar_dir: Target direction in world frame [num_envs, 2]
        tar_speed: Target speed [num_envs]
        tar_face_dir: Target facing direction in world frame [num_envs, 2]

    Returns:
        Observations [num_envs, 5]: [local_tar_dir (2), tar_speed (1), local_tar_face_dir (2)]
    """
    root_quat = root_state[:, 3:7]  # (w, x, y, z) format

    # Convert 2D directions to 3D (add zero Z component)
    tar_dir_3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[:, 0:1])], dim=-1)
    tar_face_dir_3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[:, 0:1])], dim=-1)

    # Compute heading quaternion (yaw-only rotation)
    heading_quat = calc_heading_quat_inv(root_quat)

    # Rotate target directions to local frame
    local_tar_dir = quat_rotate(heading_quat, tar_dir_3d)[:, :2]
    local_tar_face_dir = quat_rotate(heading_quat, tar_face_dir_3d)[:, :2]

    # Combine observations
    obs = torch.cat([local_tar_dir, tar_speed.unsqueeze(-1), local_tar_face_dir], dim=-1)

    return obs


@torch.jit.script
def compute_heading_reward(
    root_pos: torch.Tensor,
    prev_root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    tar_dir: torch.Tensor,
    tar_speed: torch.Tensor,
    tar_face_dir: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """Compute reward for heading-following task.

    Args:
        root_pos: Current root position [num_envs, 3]
        prev_root_pos: Previous root position [num_envs, 3]
        root_quat: Root quaternion (w,x,y,z) [num_envs, 4]
        tar_dir: Target direction [num_envs, 2]
        tar_speed: Target speed [num_envs]
        tar_face_dir: Target facing direction [num_envs, 2]
        dt: Time step

    Returns:
        Reward [num_envs]
    """
    # Reward weights
    vel_err_scale = 0.25
    tangent_err_w = 0.1
    dir_reward_w = 0.7
    facing_reward_w = 0.3

    # Compute velocity
    delta_pos = root_pos - prev_root_pos
    root_vel = delta_pos / dt

    # Project velocity onto target direction
    tar_dir_speed = torch.sum(tar_dir * root_vel[:, :2], dim=-1)

    # Compute tangential velocity (perpendicular to target direction)
    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[:, :2] - tar_dir_vel
    tangent_speed = torch.sum(tangent_vel, dim=-1)

    # Direction reward (exponential based on speed and tangent velocity errors)
    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale * (
            tar_vel_err * tar_vel_err +
            tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    # Penalize backward movement
    speed_mask = tar_dir_speed <= 0
    dir_reward[speed_mask] = 0.0

    # Facing direction reward
    heading_quat = calc_heading_quat(root_quat)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[:, 0] = 1.0  # Forward direction in body frame
    facing_dir = quat_rotate(heading_quat, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[:, :2], dim=-1)
    facing_reward = torch.clamp(facing_err, min=0.0)

    # Combined reward
    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward
