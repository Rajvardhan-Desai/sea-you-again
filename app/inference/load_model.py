"""
app/inference/load_model.py
---------------------------
Loads MARASSModel from a checkpoint and caches it as a module-level
singleton so the worker process pays the loading cost only once.

Usage
-----
    from app.inference.load_model import get_model
    model = get_model()          # subsequent calls return the same object
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger(__name__)

_model: Optional[torch.nn.Module] = None
_checkpoint_hash: Optional[str] = None

# Resolve repo root relative to this file (app/inference/load_model.py → ../../)
_REPO = Path(__file__).resolve().parent.parent.parent


def _add_model_paths() -> None:
    """Insert research module dirs into sys.path (idempotent)."""
    for sub in ("model", "model/encoders", "data-preprocessing-pipeline"):
        p = str(_REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def get_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Return the global MARASSModel singleton.

    On the first call, loads the checkpoint specified by:
      1. `checkpoint_path` argument (explicit)
      2. CHECKPOINT_PATH environment variable
      3. Fallback: <repo>/model/checkpoints/best.pt

    Subsequent calls return the cached model unless `checkpoint_path`
    differs from the one currently loaded.
    """
    global _model, _checkpoint_hash

    import os
    ckpt_str = (
        checkpoint_path
        or os.environ.get("CHECKPOINT_PATH")
        or str(_REPO / "model" / "checkpoints" / "best.pt")
    )
    ckpt_path = Path(ckpt_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    current_hash = _sha256(ckpt_path)
    if _model is not None and current_hash == _checkpoint_hash:
        return _model

    _add_model_paths()
    from model import MARASSModel, ModelConfig  # type: ignore[import]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Loading MARASSModel from {ckpt_path}  device={device}")
    cfg   = ModelConfig()
    model = MARASSModel(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    log.info(f"Checkpoint loaded — epoch={epoch}  val_loss={val_loss:.4f}  sha={current_hash}")

    _model          = model
    _checkpoint_hash = current_hash
    return _model


def get_checkpoint_hash() -> Optional[str]:
    """Return the SHA256 prefix of the currently loaded checkpoint, or None."""
    return _checkpoint_hash
