"""AMP Replay Buffer for storing demonstration and agent observations.

This is a port of old_awd/learning/replay_buffer.py to work with RSL-RL.
Used to store:
1. Demo observations: Motion capture data for discriminator training
2. Agent observations: Agent's own motion for replay buffer
"""

import torch
import numpy as np
from typing import Dict


class AMPReplayBuffer:
    """Circular replay buffer for AMP observations."""

    def __init__(self, buffer_size: int, device: str = "cpu"):
        """Initialize replay buffer.

        Args:
            buffer_size: Maximum number of samples to store.
            device: Device to store tensors on.
        """
        self.buffer_size = buffer_size
        self.device = device

        # Storage
        self.data = {}
        self.pos = 0
        self.full = False
        self.total_count = 0

    def store(self, data: Dict[str, torch.Tensor]):
        """Store data in the buffer.

        Args:
            data: Dictionary with keys like 'amp_obs' and corresponding tensors.
        """
        num_samples = None

        # Initialize storage on first call
        for key, val in data.items():
            if num_samples is None:
                num_samples = val.shape[0]

            if key not in self.data:
                # Initialize storage for this key
                sample_shape = val.shape[1:]
                self.data[key] = torch.zeros(
                    (self.buffer_size,) + sample_shape,
                    dtype=val.dtype,
                    device=self.device,
                )

        # Store data
        for key, val in data.items():
            n = val.shape[0]
            remaining = self.buffer_size - self.pos

            if n <= remaining:
                # Fits in current position
                self.data[key][self.pos : self.pos + n] = val.to(self.device)
            else:
                # Wrap around
                self.data[key][self.pos :] = val[:remaining].to(self.device)
                self.data[key][: n - remaining] = val[remaining:].to(self.device)

        # Update position
        self.pos = (self.pos + num_samples) % self.buffer_size
        self.total_count += num_samples
        if self.total_count >= self.buffer_size:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch from the buffer.

        Args:
            batch_size: Number of samples to return.

        Returns:
            Dictionary with sampled data.
        """
        # Determine valid range
        max_idx = self.buffer_size if self.full else self.pos

        # Sample random indices
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)

        # Gather samples
        batch = {}
        for key, val in self.data.items():
            batch[key] = val[indices]

        return batch

    def get_buffer_size(self) -> int:
        """Get buffer capacity.

        Returns:
            Buffer size.
        """
        return self.buffer_size

    def get_total_count(self) -> int:
        """Get total number of samples stored (including overwrites).

        Returns:
            Total count.
        """
        return self.total_count

    def __len__(self) -> int:
        """Get current number of samples in buffer.

        Returns:
            Number of valid samples.
        """
        return self.buffer_size if self.full else self.pos


class AMPDemoBuffer(AMPReplayBuffer):
    """Specialized buffer for demonstration data.

    This buffer can load motion capture data from files.
    """

    def __init__(
        self,
        buffer_size: int,
        device: str = "cpu",
        motion_files: list = None,
    ):
        """Initialize demo buffer.

        Args:
            buffer_size: Maximum number of samples to store.
            device: Device to store tensors on.
            motion_files: List of motion file paths to load.
        """
        super().__init__(buffer_size, device)
        self.motion_files = motion_files or []

    def load_motion_files(self, env):
        """Load motion data from files using environment's loader.

        Args:
            env: Environment instance with fetch_amp_obs_demo method.
        """
        if not self.motion_files:
            return

        # Use environment to load demonstrations
        # This matches the old code's approach where env handles motion loading
        for i in range(0, self.buffer_size, 512):
            batch_size = min(512, self.buffer_size - i)
            amp_obs_demo = env.fetch_amp_obs_demo(batch_size)
            self.store({"amp_obs": amp_obs_demo})

    def update_from_env(self, env, batch_size: int = 512):
        """Fetch and store new demo samples from environment.

        Args:
            env: Environment instance with fetch_amp_obs_demo method.
            batch_size: Number of samples to fetch.
        """
        amp_obs_demo = env.fetch_amp_obs_demo(batch_size)
        self.store({"amp_obs": amp_obs_demo})
