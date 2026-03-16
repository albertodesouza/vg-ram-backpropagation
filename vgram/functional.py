"""
Hard-forward / soft-backward operations for differentiable VG-RAM.

Every function here follows the STE (Straight-Through Estimator) pattern:
  forward  ->  discrete / hard value
  backward ->  gradient flows through a smooth surrogate
"""

import torch
import torch.nn.functional as F


def ste(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward returns *hard*, backward uses *soft*."""
    return soft + (hard - soft).detach()


def minchinton_compare(u: torch.Tensor, v: torch.Tensor, tau: float) -> torch.Tensor:
    """Minchinton cell comparison with STE.

    Forward : b = 1[u > v]          (hard bit)
    Backward: b ~ sigmoid((u-v)/τ)  (smooth surrogate)

    Args:
        u, v: input values being compared, any matching shape.
        tau:  temperature for the sigmoid relaxation (> 0).

    Returns:
        Tensor of same shape, values in {0, 1} on forward,
        gradients via sigmoid on backward.
    """
    diff = u - v
    hard = (diff > 0).float()
    soft = torch.sigmoid(diff / tau)
    return ste(hard, soft)


def hard_hamming(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Discrete Hamming distance (no gradient).

    Args:
        b: input bits   (..., P), values in {0, 1}
        a: stored bits  (..., P), values in {0, 1}

    Returns:
        Hamming distance summed over the last axis.
    """
    return (b != a).float().sum(dim=-1)


def expected_hamming(b_soft: torch.Tensor, a_soft: torch.Tensor) -> torch.Tensor:
    """Differentiable expected Hamming distance.

    For independent Bernoulli bits with probabilities b and a:
        E[Hamming] = Σ (b + a - 2·b·a)

    Args:
        b_soft: relaxed input bits   (..., P), values in [0, 1]
        a_soft: relaxed stored bits  (..., P), values in [0, 1]

    Returns:
        Expected Hamming distance summed over the last axis.
    """
    return (b_soft + a_soft - 2.0 * b_soft * a_soft).sum(dim=-1)


def soft_memory_select(
    d_soft: torch.Tensor,
    v_soft: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Soft memory selection via softmin (differentiable).

    Args:
        d_soft: distances (..., M) — lower is better.
        v_soft: stored values (..., M, D).
        beta:   inverse temperature for the softmin.

    Returns:
        Weighted combination of stored values (..., D).
    """
    alpha = F.softmax(-beta * d_soft, dim=-1)          # (..., M)
    return (alpha.unsqueeze(-1) * v_soft).sum(dim=-2)   # (..., D)


def hard_memory_select(
    d_hard: torch.Tensor,
    v_hard: torch.Tensor,
) -> torch.Tensor:
    """Hard memory selection: pick the entry with smallest Hamming distance.

    Args:
        d_hard: distances (..., M).
        v_hard: stored values (..., M, D).

    Returns:
        Values of the winning entry (..., D).
    """
    idx = d_hard.argmin(dim=-1)                         # (...,)
    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(
        *idx.shape, 1, v_hard.shape[-1]
    )
    return v_hard.gather(-2, idx_exp).squeeze(-2)       # (..., D)


def bits_to_scalar(bits: torch.Tensor) -> torch.Tensor:
    """Convert a vector of B bits (or soft probabilities) to a scalar in [0, 1].

    Uses positional binary weighting: Σ 2^r · bit_r  /  (2^B - 1).

    Args:
        bits: (..., B) tensor with values in {0,1} or [0,1].

    Returns:
        (...,) tensor with values in [0, 1].
    """
    B = bits.shape[-1]
    weights = (2.0 ** torch.arange(B, dtype=bits.dtype, device=bits.device))
    return (bits * weights).sum(dim=-1) / (2.0**B - 1.0)


def binary_regularization(soft_values: torch.Tensor) -> torch.Tensor:
    """Encourages soft values toward 0 or 1: R = mean(v · (1 - v))."""
    return (soft_values * (1.0 - soft_values)).mean()
