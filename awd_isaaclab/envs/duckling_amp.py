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

"""AMP (Adversarial Motion Priors) environment for quadrupedal robots.

This environment uses reference motion data to train natural-looking locomotion.
Migrated from IsaacGym's DucklingAMP to IsaacLab.
"""

from enum import Enum
import numpy as np
import torch
import os

try:
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.utils import configclass
except ImportError:
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab.utils import configclass

from .duckling_command_env import DucklingCommandEnv, DucklingCommandCfg

# Import torch_utils functions needed for dof_to_obs
try:
    from awd_isaaclab.utils import torch_utils
except ImportError:
    from ..utils import torch_utils

try:
    from awd_isaaclab.utils.bdx.amp_motion_loader import AMPLoader
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from awd_isaaclab.utils.bdx.amp_motion_loader import AMPLoader


##
# Configuration
##

@configclass
class DucklingAMPCfg(DucklingCommandCfg):
    """Configuration for AMP environment.

    Extends DucklingCommandCfg with AMP-specific parameters.
    """

    # Override num_observations with value from props file
    # AMP uses same observations as DucklingCommand for the policy
    # Separate AMP observations are used for the discriminator
    num_observations: int = 51
    """Number of observations for policy (from props file)"""

    # AMP parameters
    state_init: str = "Default"
    """State initialization method: Default, Start, Random, or Hybrid"""

    hybrid_init_prob: float = 0.5
    """Probability of using reference state init in Hybrid mode"""

    num_amp_obs_steps: int = 2
    """Number of AMP observation steps (history)"""

    motion_file: str = "awd/data/motions/go_bdx"
    """Path to motion file(s) for AMP (can be a directory or single file)"""

    key_body_names: list[str] = None
    """Names of key bodies for AMP observations (e.g., ['left_foot', 'right_foot'])"""

    local_root_obs: bool = True
    """Whether to use local root observations"""

    root_height_obs: bool = True
    """Whether to include root height in observations"""

    def __post_init__(self):
        """Override to configure observation and action spaces."""
        super().__post_init__()

        # Set default key body names if not specified
        if self.key_body_names is None:
            self.key_body_names = ["left_foot", "right_foot"]

        # AMP doesn't change standard observations for the policy
        # Separate AMP observations are handled via extras["amp_obs"]
        # observation_space and action_space are configured by parent


##
# Environment
##

class DucklingAMP(DucklingCommandEnv):
    """AMP environment for natural locomotion training.

    Uses reference motion data to train policies that produce natural-looking
    movement through adversarial motion priors.
    """

    class StateInit(Enum):
        """State initialization strategies."""
        Default = 0  # Use default initial state
        Start = 1    # Initialize from start of motion clips
        Random = 2   # Initialize from random point in motion clips
        Hybrid = 3   # Mix of default and reference initialization

    cfg: DucklingAMPCfg

    def __init__(self, cfg: DucklingAMPCfg, render_mode: str | None = None, **kwargs):
        """Initialize AMP environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
        """
        # Override num_observations BEFORE parent init
        # AMP has NO task_obs (only base duckling_obs)
        # Total: 51 (base) + 0 (task_obs) = 51
        base_obs_size = 51
        task_obs_size = 0  # AMP has no task observations
        cfg.num_observations = base_obs_size + task_obs_size

        # Parse state initialization method
        self._state_init = DucklingAMP.StateInit[cfg.state_init]
        self._hybrid_init_prob = cfg.hybrid_init_prob
        self._num_amp_obs_steps = cfg.num_amp_obs_steps
        assert self._num_amp_obs_steps >= 2, "Need at least 2 AMP observation steps"

        # Track which envs were reset with which method
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # Initialize parent
        super().__init__(cfg, render_mode, **kwargs)

        # Build key body IDs from names
        # In IsaacLab, we need to find body indices from body names
        self._build_key_body_ids(cfg.key_body_names)

        # Load robot properties for AMP observations (dof_to_obs)
        import yaml
        props_file = os.path.join(os.path.dirname(__file__), "../../awd/data/assets/go_bdx/go_bdx_props.yaml")
        props_file = os.path.abspath(props_file)
        with open(props_file, "r") as f:
            props = yaml.safe_load(f)

        self._dof_obs_size = props["dof_obs_size"]  # Number of joints
        self._dof_offsets = props["dof_offsets"]  # DOF offsets for each joint

        # Get DOF axes from robot description
        # For now, assume all 1-DOF joints with z-axis (simplified)
        num_joints = len(self._dof_offsets) - 1
        self._dof_axis_array = []
        for _ in range(num_joints):
            self._dof_axis_array.extend([0, 0, 1])  # z-axis for revolute joints

        # Load motion library
        # Convert relative path to absolute path
        motion_file = cfg.motion_file
        if not os.path.isabs(motion_file):
            motion_file = os.path.join(os.path.dirname(__file__), "../../", motion_file)
            motion_file = os.path.abspath(motion_file)

        if os.path.isdir(motion_file):
            motion_files = [
                os.path.join(motion_file, file)
                for file in os.listdir(motion_file)
                if file.endswith(".json") or file.endswith(".txt")
            ]
        else:
            motion_files = [motion_file]

        self._load_motion(motion_files)

        # Get number of AMP observations per step
        # Using exact value from props file to match original IsaacGym implementation
        self._num_amp_obs_per_step = props["num_amp_obs_per_step"]
        print(f"AMP observations per step: {self._num_amp_obs_per_step}")

        # AMP observation buffer: [num_envs, num_steps, obs_per_step]
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        # Demo buffer for discriminator training
        self._amp_obs_demo_buf = None

    def _build_key_body_ids(self, key_body_names):
        """Build key body IDs tensor from body names.

        Args:
            key_body_names: List of body names to track.
        """
        # In IsaacLab, we need to find the body indices from the robot's body names
        # The robot's body_names attribute contains all body names
        body_ids = []

        # Get all body names from the robot
        all_body_names = self._robot.data.body_names

        for body_name in key_body_names:
            try:
                body_id = all_body_names.index(body_name)
                body_ids.append(body_id)
            except ValueError:
                raise ValueError(f"Body '{body_name}' not found in robot. Available bodies: {all_body_names}")

        # Store as numpy array for AMPLoader (like original code)
        self.cfg.key_body_ids = body_ids
        print(f"Key body IDs: {body_ids} for bodies {key_body_names}")

    def _load_motion(self, motion_files):
        """Load motion library from files.

        Args:
            motion_files: List of motion file paths.
        """
        print(f"Loading motion files: {motion_files}")
        print(f"num_dof: {self.num_actions}")

        self._motion_lib = AMPLoader(
            motion_files=motion_files,
            device=self.device,
            time_between_frames=self.step_dt,
            key_body_ids=self.cfg.key_body_ids,
        )

        print(f"Loaded {self._motion_lib.num_motions()} motions")

    def get_num_amp_obs(self):
        """Get total number of AMP observations.

        Returns:
            Total AMP observation size (steps * obs_per_step)
        """
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def _post_physics_step(self):
        """Process environment after physics step."""
        # Call parent
        super()._post_physics_step()

        # Update AMP observations
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        # Add AMP observations to extras for discriminator
        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

    def fetch_amp_obs_demo(self, num_samples):
        """Fetch AMP observations from reference motions for discriminator.

        Args:
            num_samples: Number of samples to fetch.

        Returns:
            Flattened AMP observations from reference motions.
        """
        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert self._amp_obs_demo_buf.shape[0] == num_samples

        # Sample motions
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # Sample times (with truncation for history)
        truncate_time = self.step_dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 += truncate_time

        # Build AMP observations
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        """Build AMP observations from reference motions.

        Args:
            motion_ids: Motion IDs to sample from.
            motion_times0: Starting times for sampling.

        Returns:
            AMP observations from reference motions.
        """
        dt = self.step_dt

        # Expand motion IDs for all time steps
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps]
        )

        # Create time steps going backwards
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self._num_amp_obs_steps, device=self.device
        )
        motion_times = motion_times + time_steps

        # Flatten for batch processing
        motion_ids_flat = motion_ids.view(-1)
        motion_times_flat = motion_times.view(-1)

        # Get motion states
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
        ) = self._motion_lib.get_motion_state(motion_ids_flat, motion_times_flat)

        # Build AMP observations
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self.cfg.local_root_obs,
            self.cfg.root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
            self._dof_axis_array,
        )

        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        """Create buffer for demo AMP observations.

        Args:
            num_samples: Number of samples.
        """
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float32,
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: Environment IDs to reset.
        """
        # Clear reset tracking
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # Reset based on initialization method
        self._reset_actors(env_ids)

        # Call parent to handle the rest
        super()._reset_idx(env_ids)

        # Initialize AMP observations
        self._init_amp_obs(env_ids)

    def _reset_actors(self, env_ids: torch.Tensor):
        """Reset actor states based on initialization strategy.

        Args:
            env_ids: Environment IDs to reset.
        """
        if self._state_init == DucklingAMP.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == DucklingAMP.StateInit.Start
            or self._state_init == DucklingAMP.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == DucklingAMP.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            raise ValueError(
                f"Unsupported state initialization: {self._state_init}"
            )

    def _reset_default(self, env_ids: torch.Tensor):
        """Reset to default initial state.

        Args:
            env_ids: Environment IDs to reset.
        """
        num_resets = len(env_ids)

        # Reset to default position
        self._robot.write_root_pose_to_sim(
            self._robot_cfg.init_state.pos.repeat(num_resets, 1),
            self._robot_cfg.init_state.rot.repeat(num_resets, 1),
            env_ids,
        )

        # Reset velocities to zero
        self._robot.write_root_velocity_to_sim(
            torch.zeros(num_resets, 6, device=self.device),
            env_ids,
        )

        # Reset joint positions
        self._robot.write_joint_state_to_sim(
            torch.zeros(num_resets, self.num_actions, device=self.device),
            torch.zeros(num_resets, self.num_actions, device=self.device),
            env_ids,
        )

        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids: torch.Tensor):
        """Reset from reference motion data.

        Args:
            env_ids: Environment IDs to reset.
        """
        num_envs = len(env_ids)

        # Sample motions
        motion_ids = self._motion_lib.sample_motions(num_envs)

        # Sample times
        if (
            self._state_init == DucklingAMP.StateInit.Random
            or self._state_init == DucklingAMP.StateInit.Hybrid
        ):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == DucklingAMP.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            raise ValueError(f"Unsupported state init: {self._state_init}")

        # Get motion states
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times)

        # Set environment state
        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
        )

        # Track for AMP obs initialization
        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

    def _reset_hybrid_state_init(self, env_ids: torch.Tensor):
        """Reset with hybrid strategy (mix of default and reference).

        Args:
            env_ids: Environment IDs to reset.
        """
        num_envs = len(env_ids)

        # Randomly choose which envs use reference init
        ref_probs = torch.full(
            (num_envs,), self._hybrid_init_prob, device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        # Reset reference envs
        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        # Reset default envs
        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

    def _set_env_state(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        dof_pos: torch.Tensor,
        root_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        dof_vel: torch.Tensor,
    ):
        """Set environment state from motion data.

        Args:
            env_ids: Environment IDs.
            root_pos: Root positions.
            root_rot: Root rotations (quaternions).
            dof_pos: DOF positions.
            root_vel: Root linear velocities.
            root_ang_vel: Root angular velocities.
            dof_vel: DOF velocities.
        """
        num_resets = len(env_ids)

        # Set root pose
        self._robot.write_root_pose_to_sim(root_pos, root_rot, env_ids)

        # Set root velocity
        root_velocity = torch.cat([root_vel, root_ang_vel], dim=-1)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)

        # Set joint state
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids)

    def _init_amp_obs(self, env_ids: torch.Tensor):
        """Initialize AMP observations after reset.

        Args:
            env_ids: Environment IDs to initialize.
        """
        # Compute current AMP observations
        self._compute_amp_observations(env_ids)

        # Initialize history based on reset method
        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
            )

    def _init_amp_obs_default(self, env_ids: torch.Tensor):
        """Initialize AMP history by repeating current observation.

        Args:
            env_ids: Environment IDs.
        """
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs

    def _init_amp_obs_ref(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ):
        """Initialize AMP history from reference motions.

        Args:
            env_ids: Environment IDs.
            motion_ids: Motion IDs used for reset.
            motion_times: Motion times used for reset.
        """
        dt = self.step_dt

        # Create time steps for history (going backwards)
        num_envs = len(env_ids)
        motion_ids_expanded = torch.tile(
            motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1]
        )
        motion_times_expanded = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1
        )
        motion_times_hist = motion_times_expanded + time_steps

        # Flatten for batch processing
        motion_ids_flat = motion_ids_expanded.view(-1)
        motion_times_flat = motion_times_hist.view(-1)

        # Get motion states
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
        ) = self._motion_lib.get_motion_state(motion_ids_flat, motion_times_flat)

        # Build AMP observations
        amp_obs_demo = build_amp_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            self.cfg.local_root_obs,
            self.cfg.root_height_obs,
            self._dof_obs_size,
            self._dof_offsets,
            self._dof_axis_array,
        )

        # Reshape and store in history buffer
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(
            self._hist_amp_obs_buf[env_ids].shape
        )

    def _update_hist_amp_obs(self, env_ids: torch.Tensor | None = None):
        """Update AMP observation history.

        Args:
            env_ids: Environment IDs to update (None = all).
        """
        if env_ids is None:
            # Shift all history
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            # Shift history for specific envs
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]

    def _compute_amp_observations(self, env_ids: torch.Tensor | None = None):
        """Compute current AMP observations.

        Args:
            env_ids: Environment IDs to compute for (None = all).
        """
        # Get robot state
        root_state = self._robot.data.root_state_w
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        # Get key body positions
        # Note: In IsaacLab, body positions are in robot.data.body_pos_w
        # We need to get specific key bodies
        key_body_pos = self._robot.data.body_pos_w[:, self.cfg.key_body_ids, :]

        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(
                root_state[:, :3],  # root position
                root_state[:, 3:7],  # root rotation
                root_state[:, 7:10],  # root linear velocity
                root_state[:, 10:13],  # root angular velocity
                joint_pos,
                joint_vel,
                key_body_pos,
                self.cfg.local_root_obs,
                self.cfg.root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
                self._dof_axis_array,
            )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(
                root_state[env_ids, :3],
                root_state[env_ids, 3:7],
                root_state[env_ids, 7:10],
                root_state[env_ids, 10:13],
                joint_pos[env_ids],
                joint_vel[env_ids],
                key_body_pos[env_ids],
                self.cfg.local_root_obs,
                self.cfg.root_height_obs,
                self._dof_obs_size,
                self._dof_offsets,
                self._dof_axis_array,
            )


##
# JIT Compiled Helper Functions
##

@torch.jit.script
def dof_to_obs(pose: torch.Tensor, dof_obs_size: int, dof_offsets: list[int], dof_axis: list[int]) -> torch.Tensor:
    """Convert DOF positions to observations using 6D tangent-normal representation.

    This function converts joint positions (which may be 1D or 3D per joint) into a standardized
    6D representation (tangent-normal) for each joint, matching the original IsaacGym implementation.

    Args:
        pose: DOF positions [N, num_dofs]
        dof_obs_size: Number of joints (will be multiplied by 6 for output size)
        dof_offsets: Offsets for each joint in the DOF array
        dof_axis: Axis directions for each DOF (flat list of 3*num_joints values)

    Returns:
        DOF observations [N, dof_obs_size * 6]
    """
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_size_total = dof_obs_size * joint_obs_size
    dof_obs_shape = pose.shape[:-1] + (dof_obs_size_total,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device, dtype=pose.dtype)

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # Assume spherical joint (3 DOF)
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        # Single axis joint (1 DOF)
        elif dof_size == 1:
            axis = torch.tensor(
                dof_axis[j * 3:(j * 3) + 3],
                dtype=joint_pose.dtype,
                device=pose.device
            )
            joint_pose_q = torch_utils.quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            assert False, f"Unsupported joint type with {dof_size} DOFs"

        # Convert quaternion to 6D tangent-normal representation
        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert (num_joints * joint_obs_size) == dof_obs_size_total

    return dof_obs


@torch.jit.script
def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculate inverse heading quaternion (yaw-only rotation).

    Args:
        q: Quaternion (w,x,y,z) [N, 4]

    Returns:
        Inverse heading quaternion [N, 4]
    """
    # Extract yaw angle
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # Create inverse heading quaternion
    heading_quat = torch.zeros_like(q)
    heading_quat[:, 0] = torch.cos(-yaw / 2)  # w
    heading_quat[:, 3] = torch.sin(-yaw / 2)  # z

    return heading_quat


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.

    Args:
        q1: First quaternion (w,x,y,z) [N, 4]
        q2: Second quaternion (w,x,y,z) [N, 4]

    Returns:
        Product quaternion [N, 4]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion.

    Args:
        q: Quaternion (w,x,y,z) [N, 4]
        v: Vector [N, 3]

    Returns:
        Rotated vector [N, 3]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # Quaternion rotation formula
    t0 = 2.0 * (w * vx + y * vz - z * vy)
    t1 = 2.0 * (w * vy + z * vx - x * vz)
    t2 = 2.0 * (w * vz + x * vy - y * vx)

    rx = vx + x * t0 + y * t1 + z * t2
    ry = vy - y * t0 + w * t1 + x * t2
    rz = vz - z * t0 - x * t1 + w * t2

    return torch.stack([rx, ry, rz], dim=-1)


@torch.jit.script
def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to tangent-normal 6D representation.

    Args:
        q: Quaternion (w,x,y,z) [N, 4]

    Returns:
        6D representation [N, 6] (tangent + normal vectors)
    """
    # Reference tangent (x-axis)
    ref_tan = torch.zeros_like(q[:, :3])
    ref_tan[:, 0] = 1
    tan = quat_rotate(q, ref_tan)

    # Reference normal (z-axis)
    ref_norm = torch.zeros_like(q[:, :3])
    ref_norm[:, 2] = 1
    norm = quat_rotate(q, ref_norm)

    # Concatenate tangent and normal
    return torch.cat([tan, norm], dim=-1)


@torch.jit.script
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
    dof_offsets: list[int],
    dof_axis: list[int],
) -> torch.Tensor:
    """Build AMP observations from robot state.

    Args:
        root_pos: Root position [N, 3]
        root_rot: Root rotation (quaternion w,x,y,z) [N, 4]
        root_vel: Root linear velocity [N, 3]
        root_ang_vel: Root angular velocity [N, 3]
        dof_pos: DOF positions [N, num_dof]
        dof_vel: DOF velocities [N, num_dof]
        key_body_pos: Key body positions [N, num_bodies, 3]
        local_root_obs: Whether to use local root observations
        root_height_obs: Whether to include root height
        dof_obs_size: Number of joints for dof_to_obs conversion
        dof_offsets: DOF offsets for each joint
        dof_axis: DOF axis directions (flat list)

    Returns:
        AMP observations [N, obs_dim]
    """
    # Root height
    root_h = root_pos[:, 2:3]

    # Compute heading quaternion (yaw-only rotation)
    heading_rot = calc_heading_quat_inv(root_rot)

    # Root rotation observation
    if local_root_obs:
        # Rotate to local frame
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot

    # Convert to tangent-normal representation (6D)
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    # Root height observation
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    # Local root velocities
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    # Convert DOF positions to observations using 6D representation
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, dof_axis)

    # Key body positions in local frame
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    # Rotate to local frame
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))

    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    # Concatenate all observations
    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )

    return obs


@torch.jit.script
def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculate inverse heading quaternion (yaw-only rotation).

    Args:
        q: Quaternion (w,x,y,z) [N, 4]

    Returns:
        Inverse heading quaternion [N, 4]
    """
    # Extract yaw angle
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # Create inverse heading quaternion
    heading_quat = torch.zeros_like(q)
    heading_quat[:, 0] = torch.cos(-yaw / 2)  # w
    heading_quat[:, 3] = torch.sin(-yaw / 2)  # z

    return heading_quat


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.

    Args:
        q1: First quaternion (w,x,y,z) [N, 4]
        q2: Second quaternion (w,x,y,z) [N, 4]

    Returns:
        Product quaternion [N, 4]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector by quaternion.

    Args:
        q: Quaternion (w,x,y,z) [N, 4]
        v: Vector [N, 3]

    Returns:
        Rotated vector [N, 3]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # Quaternion rotation formula
    t0 = 2.0 * (w * vx + y * vz - z * vy)
    t1 = 2.0 * (w * vy + z * vx - x * vz)
    t2 = 2.0 * (w * vz + x * vy - y * vx)

    rx = vx + x * t0 + y * t1 + z * t2
    ry = vy - y * t0 + w * t1 + x * t2
    rz = vz - z * t0 - x * t1 + w * t2

    return torch.stack([rx, ry, rz], dim=-1)


@torch.jit.script
def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to tangent-normal 6D representation.

    Args:
        q: Quaternion (w,x,y,z) [N, 4]

    Returns:
        6D representation [N, 6] (tangent + normal vectors)
    """
    # Reference tangent (x-axis)
    ref_tan = torch.zeros_like(q[:, :3])
    ref_tan[:, 0] = 1
    tan = quat_rotate(q, ref_tan)

    # Reference normal (z-axis)
    ref_norm = torch.zeros_like(q[:, :3])
    ref_norm[:, 2] = 1
    norm = quat_rotate(q, ref_norm)

    # Concatenate
    norm_tan = torch.cat([tan, norm], dim=-1)
    return norm_tan
