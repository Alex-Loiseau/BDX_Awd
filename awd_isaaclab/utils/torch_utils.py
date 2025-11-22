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

"""Torch utilities for IsaacLab.

Migrated from IsaacGym's torch_utils to work with IsaacLab.
Provides quaternion and rotation utilities.
"""

import torch
import numpy as np
from typing import Tuple
from torch import Tensor

# Import base quaternion functions from IsaacLab or define them
try:
    from isaaclab.utils.math import (
        quat_rotate,
        quat_mul,
        quat_conjugate,
        quat_from_euler_xyz,
        quat_from_angle_axis,
        normalize_angle,
    )
except ImportError:
    # Fallback: define basic functions if IsaacLab doesn't have them
    # These are standard quaternion operations

    @torch.jit.script
    def quat_conjugate(q):
        # type: (Tensor) -> Tensor
        """Compute quaternion conjugate."""
        return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

    @torch.jit.script
    def quat_mul(q1, q2):
        # type: (Tensor, Tensor) -> Tensor
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[..., 3], q1[..., 0], q1[..., 1], q1[..., 2]
        w2, x2, y2, z2 = q2[..., 3], q2[..., 0], q2[..., 1], q2[..., 2]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.stack([x, y, z, w], dim=-1)

    @torch.jit.script
    def quat_rotate(q, v):
        # type: (Tensor, Tensor) -> Tensor
        """Rotate vector by quaternion."""
        # q is (x, y, z, w) format
        # v is a 3D vector
        shape = q.shape
        q_w = q[..., 3]
        q_vec = q[..., :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    @torch.jit.script
    def quat_from_euler_xyz(roll, pitch, yaw):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        """Create quaternion from Euler angles (roll, pitch, yaw)."""
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        return torch.stack([qx, qy, qz, qw], dim=-1)

    @torch.jit.script
    def quat_from_angle_axis(angle, axis):
        # type: (Tensor, Tensor) -> Tensor
        """Create quaternion from angle-axis representation."""
        theta = angle / 2.0
        xyz = axis * torch.sin(theta).unsqueeze(-1)
        w = torch.cos(theta)
        return torch.cat([xyz, w.unsqueeze(-1)], dim=-1)

    @torch.jit.script
    def normalize_angle(x):
        # type: (Tensor) -> Tensor
        """Normalize angle to [-pi, pi]."""
        return torch.atan2(torch.sin(x), torch.cos(x))


# Additional quaternion/rotation utilities
@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    """Compute axis-angle representation from quaternion q.

    Args:
        q: Quaternion in (x, y, z, w) format (must be normalized)

    Returns:
        Tuple of (angle, axis)
    """
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(torch.clamp(q[..., qw], -1.0, 1.0))
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / (sin_theta_expand + 1e-8)  # Add epsilon for numerical stability

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    """Compute exponential map from axis-angle."""
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


@torch.jit.script
def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    """Compute exponential map from quaternion.

    Args:
        q: Quaternion (must be normalized)

    Returns:
        Exponential map representation
    """
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


@torch.jit.script
def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    """Represent a rotation using tangent and normal vectors.

    Args:
        q: Quaternion

    Returns:
        6D rotation representation (tangent + normal)
    """
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Convert Euler angles to exponential map."""
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map


@torch.jit.script
def exp_map_to_angle_axis(exp_map):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    """Convert exponential map to angle-axis representation."""
    min_theta = 1e-5

    angle = torch.norm(exp_map, dim=-1)
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map / (angle_exp + 1e-8)  # Add epsilon for numerical stability
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


@torch.jit.script
def exp_map_to_quat(exp_map):
    # type: (Tensor) -> Tensor
    """Convert exponential map to quaternion."""
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q


@torch.jit.script
def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: Starting quaternion
        q1: Ending quaternion
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    # Ensure shortest path
    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(torch.clamp(cos_half_theta, -1.0, 1.0))
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / (sin_half_theta + 1e-8)
    ratioB = torch.sin(t * half_theta) / (sin_half_theta + 1e-8)

    new_q = ratioA * q0 + ratioB * q1

    # Handle special cases
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


@torch.jit.script
def calc_heading(q):
    # type: (Tensor) -> Tensor
    """Calculate heading direction from quaternion.

    The heading is the direction on the xy plane.

    Args:
        q: Quaternion (must be normalized)

    Returns:
        Heading angle in radians
    """
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


@torch.jit.script
def calc_heading_quat(q):
    # type: (Tensor) -> Tensor
    """Calculate heading rotation quaternion from quaternion.

    The heading is the rotation around the z-axis only.

    Args:
        q: Quaternion (must be normalized)

    Returns:
        Heading quaternion (rotation around z-axis only)
    """
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


@torch.jit.script
def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    """Calculate inverse heading rotation quaternion from quaternion.

    The heading is the rotation around the z-axis only.

    Args:
        q: Quaternion (must be normalized)

    Returns:
        Inverse heading quaternion
    """
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q
