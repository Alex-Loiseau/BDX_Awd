"""Duckling Command environment - IsaacLab version.

Migration of awd/env/tasks/duckling_command.py from IsaacGym to IsaacLab.
This environment trains the robot to follow velocity commands (vx, vy, vyaw).
"""

import torch
import numpy as np
from typing import Dict, Any
import gymnasium as gym
import omni.usd

# IsaacLab 0.48.4+ uses 'isaaclab' namespace
try:
    from isaaclab.utils.math import quat_rotate_inverse
    from isaaclab.utils.configclass import configclass
    from isaaclab.assets import ArticulationCfg, Articulation
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
except ImportError:
    from omni.isaac.lab.utils.math import quat_rotate_inverse
    from omni.isaac.lab.utils.configclass import configclass
    from omni.isaac.lab.assets import ArticulationCfg, Articulation
    from omni.isaac.lab.sim.spawners.materials import RigidBodyMaterialCfg

from .duckling_base_env import DucklingBaseEnv, DucklingBaseCfg


@configclass
class DucklingCommandCfg(DucklingBaseCfg):
    """Configuration for DucklingCommand environment."""

    # Command ranges
    command_x_range: tuple = (-0.3, 0.3)  # m/s
    command_y_range: tuple = (-0.3, 0.3)  # m/s
    command_yaw_range: tuple = (-0.2, 0.2)  # rad/s

    # Normalization scales
    lin_vel_scale: float = 0.5
    ang_vel_scale: float = 0.25

    # Use average velocities for smoother training
    use_average_velocities: bool = True
    velocity_averaging_window: int = 10

    # Reward scales (will be multiplied by dt in __init__)
    lin_vel_xy_reward_scale: float = 0.5
    ang_vel_z_reward_scale: float = 0.25
    torque_reward_scale: float = -0.000025
    action_rate_reward_scale: float = 0.0  # From IsaacGym config - no action rate penalty
    stand_still_reward_scale: float = 0.0

    # Robot asset (to be set by specific robot config)
    robot: ArticulationCfg = None

    # Keyboard input for manual control
    keyboard_input: bool = False

    # Enable task observations (commands in obs)
    enable_task_obs: bool = True


class DucklingCommandEnv(DucklingBaseEnv):
    """Environment for training robot to follow velocity commands.

    The robot learns to track commanded velocities (vx, vy, vyaw) while
    maintaining balance and minimizing energy consumption.

    Observations:
        - Root orientation, linear/angular velocities
        - Joint positions and velocities
        - Previous actions
        - Velocity commands (task obs)

    Actions:
        - Joint torques (or PD targets)

    Rewards:
        - Tracking linear velocity (xy)
        - Tracking angular velocity (yaw)
        - Torque penalty
        - Action rate penalty
        - Stand still penalty when commands are zero
    """

    cfg: DucklingCommandCfg

    def __init__(self, cfg: DucklingCommandCfg, render_mode: str | None = None, **kwargs):
        """Initialize DucklingCommand environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for visualization.
        """
        # Validate configuration
        if cfg.robot is None:
            raise ValueError("Robot configuration must be provided!")

        # Calculate total observation size: base (51) + task_obs (3 if enabled)
        # Base duckling_obs: 51 = projected_gravity (3) + dof_pos (16) + dof_vel (16) + prev_actions (16)
        # Task obs: 3 = commands_scaled (3)
        base_obs_size = 51
        task_obs_size = 3 if cfg.enable_task_obs else 0
        cfg.num_observations = base_obs_size + task_obs_size

        # Define observation and action spaces before parent init
        # These will be validated before the environment is created
        if cfg.num_observations > 0:
            cfg.observation_space = gym.spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(cfg.num_observations,), dtype=np.float32
            )
        if cfg.num_actions > 0:
            cfg.action_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=(cfg.num_actions,), dtype=np.float32
            )

        # Store reward scales (will multiply by dt after parent init)
        self._raw_reward_scales = {
            "lin_vel_xy": cfg.lin_vel_xy_reward_scale,
            "ang_vel_z": cfg.ang_vel_z_reward_scale,
            "torque": cfg.torque_reward_scale,
            "action_rate": cfg.action_rate_reward_scale,
            "stand_still": cfg.stand_still_reward_scale,
        }

        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)

        # Scale torque and action rate rewards by dt (as in original code)
        self.rew_scales = self._raw_reward_scales.copy()
        self.rew_scales["torque"] *= self.dt
        self.rew_scales["action_rate"] *= self.dt

        # Velocity commands (vx, vy, vyaw)
        self.commands = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self.commands_scale = torch.tensor(
            [self.cfg.lin_vel_scale, self.cfg.lin_vel_scale, self.cfg.ang_vel_scale],
            device=self.device,
            dtype=torch.float32,
        )

        # Average velocities for smoother training
        if self.cfg.use_average_velocities:
            self.velocity_history = torch.zeros(
                (self.num_envs, self.cfg.velocity_averaging_window, 6),
                device=self.device,
                dtype=torch.float32,
            )
            self.velocity_history_index = 0

        # Keyboard control setup
        if self.cfg.keyboard_input:
            try:
                import pygame
                pygame.init()
                self._screen = pygame.display.set_mode((100, 100))
                pygame.display.set_caption("Arrow keys to move robot")
                print("[DucklingCommand] Keyboard control enabled!")
                print("  Z/S: Forward/Backward")
                print("  Q/D: Left/Right")
                print("  A/E: Turn Left/Right")
            except ImportError:
                print("[Warning] pygame not installed, keyboard input disabled")
                self.cfg.keyboard_input = False

    def _setup_scene(self) -> None:
        """Setup the simulation scene with robot and terrain.

        Note: Ground plane is included in the USD file (added manually in Isaac Sim).
        No need to create it programmatically.
        """
        print("[DEBUG ENV] _setup_scene() called")
        # Add robot to scene
        self._robot_cfg = self.cfg.robot
        self._robot = Articulation(self._robot_cfg)
        self.scene.articulations["robot"] = self._robot
        print(f"[DEBUG ENV] Robot created: {self._robot}")

        # Clone scene
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions (if needed)
        self.scene.filter_collisions(global_prim_paths=[])

        # Note: Lights are created automatically by Isaac Sim when viewing in GUI

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions and update commands.

        Args:
            actions: Actions from policy.
        """
        # Update commands from keyboard if enabled
        if self.cfg.keyboard_input:
            self._update_commands_from_keyboard()

        # Apply pushes if enabled
        self._apply_random_pushes()

        # Call parent to apply actions
        super()._pre_physics_step(actions)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations.

        Returns:
            Dictionary with "policy" observations.
        """
        # Get robot state
        root_quat = self._robot.data.root_quat_w  # (w, x, y, z)
        root_lin_vel = self._robot.data.root_lin_vel_w
        root_ang_vel = self._robot.data.root_ang_vel_w
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel

        # Transform velocities to local frame
        base_lin_vel = quat_rotate_inverse(root_quat, root_lin_vel)
        base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel)

        # Update velocity history for averaging
        if self.cfg.use_average_velocities:
            self.velocity_history[:, self.velocity_history_index] = torch.cat(
                [root_lin_vel, root_ang_vel], dim=-1
            )
            self.velocity_history_index = (
                self.velocity_history_index + 1
            ) % self.cfg.velocity_averaging_window

            # Compute average velocities
            avg_lin_vel = self.velocity_history[:, :, :3].mean(dim=1)
            avg_ang_vel = self.velocity_history[:, :, 3:].mean(dim=1)

            # Transform to local frame
            avg_base_lin_vel = quat_rotate_inverse(root_quat, avg_lin_vel)
            avg_base_ang_vel = quat_rotate_inverse(root_quat, avg_ang_vel)
        else:
            avg_base_lin_vel = base_lin_vel
            avg_base_ang_vel = base_ang_vel

        # Compute projected gravity (Z-axis of root orientation in world frame)
        # This gives us gravity direction in local frame - matching original IsaacGym
        gravity_vec = torch.tensor([0., 0., -1.], device=self.device, dtype=torch.float32)
        projected_gravity = quat_rotate_inverse(root_quat, gravity_vec.unsqueeze(0).expand(root_quat.shape[0], -1))

        # Build base duckling observations - matching original IsaacGym
        # Base obs: projected_gravity, dof_pos, dof_vel, prev_actions = 51 dims
        duckling_obs = torch.cat(
            [
                projected_gravity,               # 3
                dof_pos,                         # num_dof (16)
                dof_vel,                         # num_dof (16)
                self.prev_actions,               # num_actions (16)
            ],
            dim=-1,
        )

        # Add task observations (commands) if enabled
        # Task obs: commands_scaled = 3 dims
        # Total: 51 + 3 = 54 dimensions
        if self.cfg.enable_task_obs:
            task_obs = self.commands * self.commands_scale  # 3
            obs = torch.cat([duckling_obs, task_obs], dim=-1)
        else:
            obs = duckling_obs

        # Store for reward computation
        self._avg_base_lin_vel = avg_base_lin_vel
        self._avg_base_ang_vel = avg_base_ang_vel

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards.

        Returns:
            Reward tensor for each environment.
        """
        # Use averaged velocities if enabled
        if self.cfg.use_average_velocities:
            base_lin_vel = self._avg_base_lin_vel
            base_ang_vel = self._avg_base_ang_vel
        else:
            root_quat = self._robot.data.root_quat_w
            base_lin_vel = quat_rotate_inverse(root_quat, self._robot.data.root_lin_vel_w)
            base_ang_vel = quat_rotate_inverse(root_quat, self._robot.data.root_ang_vel_w)

        # Linear velocity tracking (xy only)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1
        )
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]

        # Angular velocity tracking (yaw only)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # Torque penalty
        torques = self._robot.data.applied_torque
        rew_torque = torch.sum(torch.square(torques), dim=1) * self.rew_scales["torque"]

        # Action rate penalty
        rew_action_rate = (
            torch.sum(torch.square(self.prev_actions - self.actions), dim=1)
            * self.rew_scales["action_rate"]
        )

        # Stand still penalty (penalize motion when no command)
        command_norm = torch.norm(self.commands, dim=1)
        dof_pos_error = torch.sum(
            torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos),
            dim=1,
        )
        rew_stand_still = (
            dof_pos_error * (command_norm < 0.01) * self.rew_scales["stand_still"]
        )

        # Total reward (clipped to be non-negative)
        total_reward = (
            rew_lin_vel_xy
            + rew_ang_vel_z
            + rew_torque
            + rew_action_rate
            + rew_stand_still
        )
        total_reward = torch.clamp(total_reward, min=0.0)

        return total_reward

    def _reset_task(self, env_ids: torch.Tensor) -> None:
        """Reset task-specific state (randomize commands).

        Args:
            env_ids: Environment indices to reset.
        """
        # Randomize commands
        self.commands[env_ids, 0] = torch.rand(
            len(env_ids), device=self.device
        ) * (self.cfg.command_x_range[1] - self.cfg.command_x_range[0]) + self.cfg.command_x_range[0]

        self.commands[env_ids, 1] = torch.rand(
            len(env_ids), device=self.device
        ) * (self.cfg.command_y_range[1] - self.cfg.command_y_range[0]) + self.cfg.command_y_range[0]

        self.commands[env_ids, 2] = torch.rand(
            len(env_ids), device=self.device
        ) * (self.cfg.command_yaw_range[1] - self.cfg.command_yaw_range[0]) + self.cfg.command_yaw_range[0]

        # Reset velocity history
        if self.cfg.use_average_velocities:
            self.velocity_history[env_ids] = 0.0

    def _update_commands_from_keyboard(self) -> None:
        """Update commands from keyboard input."""
        try:
            import pygame

            keys = pygame.key.get_pressed()

            lin_vel_x = 0.0
            lin_vel_y = 0.0
            ang_vel = 0.0

            if keys[pygame.K_z]:
                lin_vel_x = self.cfg.command_x_range[1]
            if keys[pygame.K_s]:
                lin_vel_x = self.cfg.command_x_range[0]
            if keys[pygame.K_d]:
                lin_vel_y = self.cfg.command_y_range[0]
            if keys[pygame.K_q]:
                lin_vel_y = self.cfg.command_y_range[1]
            if keys[pygame.K_a]:
                ang_vel = self.cfg.command_yaw_range[1]
            if keys[pygame.K_e]:
                ang_vel = self.cfg.command_yaw_range[0]

            self.commands[:, 0] = lin_vel_x
            self.commands[:, 1] = lin_vel_y
            self.commands[:, 2] = ang_vel

            pygame.event.pump()

        except ImportError:
            pass
