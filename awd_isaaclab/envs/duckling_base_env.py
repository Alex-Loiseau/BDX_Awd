"""Base environment for Duckling locomotion tasks - IsaacLab version.

This is the IsaacLab migration of awd/env/tasks/duckling.py
Replaces IsaacGym's BaseTask with IsaacLab's DirectRLEnv.
"""

import torch
import numpy as np
from typing import Dict, Any
import gymnasium as gym
from dataclasses import MISSING

# IsaacLab 0.48.4+ uses 'isaaclab' namespace instead of 'omni.isaac.lab'
try:
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, PhysxCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils.math import quat_rotate_inverse, quat_mul, quat_conjugate
    from isaaclab.utils.configclass import configclass
except ImportError:
    # Fallback for older versions
    from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
    from omni.isaac.lab.assets import Articulation
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
    from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg
    from omni.isaac.lab.terrains import TerrainImporterCfg
    from omni.isaac.lab.utils.math import quat_rotate_inverse, quat_mul, quat_conjugate
    from omni.isaac.lab.utils.configclass import configclass


@configclass
class DucklingBaseCfg(DirectRLEnvCfg):
    """Base configuration for Duckling environments."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
        replicate_physics=True,
    )

    # Episode settings
    episode_length_s: float = 500 * (1.0 / 60.0)  # 500 steps @ 60Hz

    # Decimation: Number of control steps per simulation step
    decimation: int = 2  # Equivalent to controlFrequencyInv

    # Observations and actions (to be set by subclasses)
    observation_space: gym.Space = MISSING  # Will be set in __init__
    num_observations: int = -1
    action_space: gym.Space = MISSING  # Will be set in __init__
    num_actions: int = -1
    num_states: int = 0

    # Control settings
    pd_control: str = "custom"  # "isaac" or "custom"
    power_scale: float = 1.0
    action_scale: float = 1.0

    # Termination settings
    enable_early_termination: bool = True
    termination_height: float = 0.1
    head_termination_height: float = 0.3

    # Observation settings
    local_root_obs: bool = True
    root_height_obs: bool = True
    enable_task_obs: bool = True

    # Randomization
    randomize_com: bool = False
    com_range: list = None
    randomize_torques: bool = False
    torque_multiplier_range: list = None

    # Push settings
    push_robots: bool = False
    push_interval_s: float = 2.0
    max_push_vel_xy: float = 0.3

    # Debug
    debug_vis: bool = False


class DucklingBaseEnv(DirectRLEnv):
    """Base environment for Duckling locomotion tasks.

    This class provides the core functionality for quadrupedal locomotion tasks,
    including observation computation, reward calculation, and reset logic.

    Migrated from IsaacGym's Duckling class to IsaacLab's DirectRLEnv.
    """

    cfg: DucklingBaseCfg

    def __init__(self, cfg: DucklingBaseCfg, render_mode: str | None = None, **kwargs):
        """Initialize the Duckling environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
        """
        # Store configuration
        self.cfg = cfg

        # Robot reference (will be set by _setup_scene during parent init)
        self._robot: Articulation | None = None

        # Initialize parent class (this will call _setup_scene which sets self._robot)
        super().__init__(cfg, render_mode, **kwargs)

        # Store num_actions and num_observations as attributes for compatibility
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        # Counters
        self.common_step_counter = 0
        self.push_interval = int(np.ceil(self.cfg.push_interval_s / self.dt))

        # Previous actions for action rate penalty
        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=torch.float32,
        )

        # Termination buffer
        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Randomization buffers
        if self.cfg.randomize_com:
            self.com_offsets = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=torch.float32
            )
            self._randomize_com()

        if self.cfg.randomize_torques:
            self.torque_multipliers = torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
            )
            self._randomize_torques()

    # Note: _setup_scene() is NOT defined here - it must be implemented by subclasses
    # This allows DirectRLEnv to call the subclass implementation directly

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step.

        Args:
            actions: Actions from the policy.
        """
        # Store previous actions
        self.prev_actions[:] = self.actions

        # Clip and scale actions
        self.actions = torch.clamp(actions, -1.0, 1.0)
        scaled_actions = self.actions * self.cfg.action_scale

        # Apply torque randomization if enabled
        if self.cfg.randomize_torques:
            scaled_actions = scaled_actions * self.torque_multipliers

        # Apply actions to robot
        if self.cfg.pd_control == "custom":
            # Custom PD control (to be implemented in subclass if needed)
            self._apply_custom_pd_control(scaled_actions)
        else:
            # Use Isaac's built-in actuators
            self._robot.set_joint_effort_target(scaled_actions)

    def _apply_custom_pd_control(self, actions: torch.Tensor) -> None:
        """Apply custom PD control.

        This can be overridden by subclasses for custom control.
        Default implementation just applies effort directly.

        Args:
            actions: Scaled actions to apply.
        """
        self._robot.set_joint_effort_target(actions * self.cfg.power_scale)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations.

        Must be implemented by subclasses.

        Returns:
            Dictionary with observation data.
        """
        raise NotImplementedError("Subclasses must implement _get_observations")

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards.

        Must be implemented by subclasses.

        Returns:
            Reward tensor for each environment.
        """
        raise NotImplementedError("Subclasses must implement _get_rewards")

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute done flags.

        Returns:
            Tuple of (dones, time_outs) tensors.
        """
        # Check if robot has fallen
        if self.cfg.enable_early_termination:
            root_pos = self._robot.data.root_pos_w
            fallen = root_pos[:, 2] < self.cfg.termination_height

            # Check head height if needed
            if hasattr(self, '_head_body_id'):
                head_height = self._robot.data.body_pos_w[:, self._head_body_id, 2]
                fallen = fallen | (head_height < self.cfg.head_termination_height)
        else:
            fallen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Check time limit
        time_out = self.episode_length_buf >= self.max_episode_length

        # Combine
        dones = fallen | time_out

        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        if len(env_ids) == 0:
            return

        # Reset robot state
        self._reset_robot_state(env_ids)

        # Reset previous actions
        self.prev_actions[env_ids] = 0.0

        # Apply randomizations if needed
        if self.cfg.randomize_com:
            self._randomize_com(env_ids)

        if self.cfg.randomize_torques:
            self._randomize_torques(env_ids)

        # Reset task-specific state (can be overridden)
        self._reset_task(env_ids)

        # Clear buffers
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

    def _reset_robot_state(self, env_ids: torch.Tensor) -> None:
        """Reset robot to initial state.

        Args:
            env_ids: Environment indices to reset.
        """
        # Get default state
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # Add noise if needed (can be customized in subclass)
        # For now, just use defaults

        # Set root state
        self._robot.write_root_pose_to_sim(
            default_root_state[:, :7], env_ids=env_ids
        )
        self._robot.write_root_velocity_to_sim(
            default_root_state[:, 7:], env_ids=env_ids
        )

        # Set joint state
        self._robot.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, env_ids=env_ids
        )

    def _reset_task(self, env_ids: torch.Tensor) -> None:
        """Reset task-specific state.

        Can be overridden by subclasses.

        Args:
            env_ids: Environment indices to reset.
        """
        pass

    def _randomize_com(self, env_ids: torch.Tensor | None = None) -> None:
        """Randomize center of mass.

        Args:
            env_ids: Environment indices to randomize. If None, randomize all.
        """
        if not self.cfg.randomize_com or self.cfg.com_range is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Generate random COM offsets
        com_min, com_max = self.cfg.com_range
        self.com_offsets[env_ids] = torch.rand(
            (len(env_ids), 3), device=self.device
        ) * (com_max - com_min) + com_min

        # Note: Actual COM modification would require modifying rigid body properties
        # This is more complex in IsaacLab and may need to be done at spawn time

    def _randomize_torques(self, env_ids: torch.Tensor | None = None) -> None:
        """Randomize torque multipliers.

        Args:
            env_ids: Environment indices to randomize. If None, randomize all.
        """
        if not self.cfg.randomize_torques or self.cfg.torque_multiplier_range is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Generate random torque multipliers
        mult_min, mult_max = self.cfg.torque_multiplier_range
        self.torque_multipliers[env_ids] = torch.rand(
            (len(env_ids), self.num_actions), device=self.device
        ) * (mult_max - mult_min) + mult_min

    def _apply_random_pushes(self) -> None:
        """Apply random pushes to robots at intervals."""
        if not self.cfg.push_robots:
            return

        self.common_step_counter += 1
        if self.common_step_counter % self.push_interval != 0:
            return

        # Generate random push velocities
        push_vels = torch.rand(
            (self.num_envs, 2), device=self.device
        ) * 2.0 - 1.0  # [-1, 1]
        push_vels *= self.cfg.max_push_vel_xy

        # Apply to root
        root_vel = self._robot.data.root_lin_vel_w.clone()
        root_vel[:, :2] += push_vels

        self._robot.write_root_velocity_to_sim(
            torch.cat([root_vel, self._robot.data.root_ang_vel_w], dim=-1)
        )

    # Utility methods for quaternion operations
    def _quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse of quaternion.

        Args:
            q: Quaternion (w, x, y, z).
            v: Vector to rotate.

        Returns:
            Rotated vector.
        """
        return quat_rotate_inverse(q, v)

    @property
    def dt(self) -> float:
        """Timestep of the environment."""
        return self.cfg.sim.dt * self.cfg.decimation
