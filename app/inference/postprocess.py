"""
app/inference/postprocess.py
----------------------------
Convert raw model outputs (logits / log-variance) to calibrated quantities
and call compute_ecosystem_impact from model.model.

All functions are pure (no side effects, no I/O).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent

def _ensure_model_path() -> None:
    p = str(_REPO / "model")
    if p not in sys.path:
        sys.path.insert(0, p)


def bloom_probs(bloom_logits: torch.Tensor) -> torch.Tensor:
    """sigmoid(bloom_logits) → (B, 5, H, W) in [0, 1]."""
    return torch.sigmoid(bloom_logits.float())


def eri_classes(eri_logits: torch.Tensor) -> torch.Tensor:
    """argmax over class dim → (B, H, W) integer 0-4."""
    return eri_logits.argmax(dim=2 if eri_logits.dim() == 5 else 1).long()


def uncertainty_std(log_var: torch.Tensor) -> torch.Tensor:
    """exp(log_var / 2) → aleatoric standard deviation, same shape."""
    return (log_var.float() * 0.5).exp()


def ecosystem_impact(
    bloom_logits: torch.Tensor,   # (B, 5, H, W)
    forecast:     torch.Tensor,   # (B, 5, H, W)
    log_var:      torch.Tensor,   # (B, 1, H, W)
    static:       torch.Tensor,   # (B, C, H, W)
    land_mask:    torch.Tensor,   # (B, H, W)
) -> torch.Tensor:
    """
    Wrapper around model.model.compute_ecosystem_impact.

    Returns (B, H, W) scores in [0, 1].
    """
    _ensure_model_path()
    from model import compute_ecosystem_impact  # type: ignore[import]

    b_probs = bloom_probs(bloom_logits)
    return compute_ecosystem_impact(b_probs, forecast, log_var, static, land_mask)
