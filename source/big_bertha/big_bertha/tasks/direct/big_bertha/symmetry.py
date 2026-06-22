# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Mirror-symmetry data augmentation for Big Bertha (forward / x-axis mirror).

The PhysX->DART sim-to-sim crab is a *lateral* slip: the gait, optimal in PhysX,
pushes the body sideways under DART's contact solver. A gait that is exactly
mirror-symmetric about the forward (x) axis has no inherent lateral bias, so it
walks straight in ANY contact engine -- which is what kills the crab at the
source (rather than compensating for it). This module reflects the 48-D
observation and 12-D action about the x-z plane (y -> -y) for rsl_rl's symmetry
augmentation (Mittal 2024).

Robot geometry (from the URDF), legs in the type-grouped joint order
hips[110,113,116,119] knees[111,114,117,120] ankles[112,115,118,121]:
  leg0(110)=back-right  leg1(113)=front-right  leg2(116)=front-left  leg3(119)=back-left
Under the y->-y reflection the legs pair 0<->3 and 1<->2. Joint axes are
pseudovectors (transform as (-ax, ay, -az)): the hip z-axis flips -> hip angle
sign flips; the knee/ankle axes map onto the paired leg's axis exactly -> no
sign flip.

obs layout (48): lin_vel_b(3) ang_vel_b(3) projected_gravity(3) commands(3)
                 joint_pos(12) joint_vel(12) prev_actions(12)
"""

from __future__ import annotations

import torch

# joint mirror: leg0<->3, leg1<->2 within each of hips/knees/ankles
_PERM = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
# hips flip sign (z-axis pseudovector), knees + ankles do not
_SIGN = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

_perm_t: torch.Tensor | None = None
_sign_t: torch.Tensor | None = None


def _mirror_joints(x: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Mirror a [..., 12] joint tensor: permute legs then flip hip signs."""
    return x[..., perm] * sign


def _mirror_obs(o: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Reflect the 48-D policy observation about the forward (x) axis."""
    m = o.clone()
    m[..., 1] = -o[..., 1]  # lin_vel y
    m[..., 3] = -o[..., 3]
    m[..., 5] = -o[..., 5]  # ang_vel x, z
    m[..., 7] = -o[..., 7]  # gravity y
    m[..., 10] = -o[..., 10]
    m[..., 11] = -o[..., 11]  # commands vy, wz
    m[..., 12:24] = _mirror_joints(o[..., 12:24], perm, sign)  # joint_pos
    m[..., 24:36] = _mirror_joints(o[..., 24:36], perm, sign)  # joint_vel
    m[..., 36:48] = _mirror_joints(o[..., 36:48], perm, sign)  # prev_actions
    return m


def compute_symmetry(env=None, obs=None, actions=None, action=None, obs_type: str = "policy"):
    """rsl_rl data-augmentation hook: return (orig||mirror) for obs and actions.

    Either obs or actions may be None (rsl_rl calls it both ways). obs is a
    TensorDict ({"policy": [B, 48]}); actions is a plain [B, 12] tensor. The
    mirror is an involution (its own inverse).
    """
    if action is not None and actions is None:
        actions = action
    ref = obs if obs is not None else actions
    dev = (ref if torch.is_tensor(ref) else ref["policy"]).device

    global _perm_t, _sign_t
    if _perm_t is None or _perm_t.device != dev:
        _perm_t = torch.tensor(_PERM, dtype=torch.long, device=dev)
        _sign_t = torch.tensor(_SIGN, dtype=torch.float, device=dev)
    perm, sign = _perm_t, _sign_t

    out_obs = None
    if obs is not None:
        if torch.is_tensor(obs):
            out_obs = torch.cat([obs, _mirror_obs(obs, perm, sign)], dim=0)
        else:
            # TensorDict / dict: mirror the policy obs, keep batch_size = 2B
            o = obs["policy"]
            aug = torch.cat([o, _mirror_obs(o, perm, sign)], dim=0)
            try:
                from tensordict import TensorDict

                out_obs = TensorDict({"policy": aug}, batch_size=[aug.shape[0]], device=aug.device)
                for k in obs.keys():  # carry any other obs groups (duplicated)
                    if k != "policy":
                        v = obs[k]
                        out_obs[k] = torch.cat([v, v], dim=0)
            except Exception:
                out_obs = {"policy": aug}

    out_actions = None
    if actions is not None:
        out_actions = torch.cat([actions, _mirror_joints(actions, perm, sign)], dim=0)

    return out_obs, out_actions
