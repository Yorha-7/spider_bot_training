"""Convert measured lower-leg contact forces to four digital switch readings."""

import torch

SWITCH_THRESHOLD_N = 0.98


def foot_switches(foot_forces_w: torch.Tensor) -> torch.Tensor:
    """Return FR, FL, RL, RR switch states from foot-contact forces in newtons.

    The final dimensions of the input are (4, 3); preceding dimensions are
    batches. The result contains floating-point zeroes and ones for the policy.
    """
    if foot_forces_w.shape[-2:] != (4, 3):
        raise ValueError("foot_forces_w must have shape (..., 4, 3)")
    return (torch.linalg.vector_norm(foot_forces_w, dim=-1) >= SWITCH_THRESHOLD_N).to(foot_forces_w.dtype)
