"""
app/inference/render.py
-----------------------
Convert numpy arrays → PNG overlay images written to
  <run_dir>/overlays/<layer>_d<horizon>.png

Also writes a colorbar PNG per layer and a metadata.json.

Usage
-----
    from app.inference.render import render_run_overlays
    render_run_overlays(run_dir, arrays, bbox, horizons)
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Layer rendering config
# ──────────────────────────────────────────────────────────────────────────────

class LayerCfg(NamedTuple):
    cmap: str
    vmin: float | None  # None → per-run percentile
    vmax: float | None
    log_scale: bool = False  # log1p before mapping
    n_levels: int | None = None  # for discrete colormaps (ERI)


LAYER_CFG: dict[str, LayerCfg] = {
    "recon":    LayerCfg("viridis", None, None, log_scale=True),
    "forecast": LayerCfg("viridis", None, None, log_scale=True),
    "bloom":    LayerCfg("YlOrRd",  0.0,  1.0),
    "eri":      LayerCfg("RdYlGn_r", 0, 4, n_levels=5),
    "impact":   LayerCfg("magma",   0.0,  1.0),
}

# ERI discrete palette (level 0=green .. 4=dark red)
_ERI_COLORS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _render_layer(
    arr: np.ndarray,        # (H, W)
    land_mask: np.ndarray,  # (H, W) — 1 = land
    cfg: LayerCfg,
    out_path: Path,
) -> None:
    plt = _plt()
    import matplotlib.colors as mcolors

    data = arr.copy().astype(float)
    data[land_mask.astype(bool)] = np.nan

    if cfg.log_scale:
        data = np.log1p(np.clip(data, 0, None))

    vmin = cfg.vmin if cfg.vmin is not None else float(np.nanpercentile(data, 2))
    vmax = cfg.vmax if cfg.vmax is not None else float(np.nanpercentile(data, 98))
    if cfg.n_levels:
        vmin, vmax = 0, cfg.n_levels - 1

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    if cfg.n_levels:
        cmap = mcolors.ListedColormap(_ERI_COLORS)
        norm = mcolors.BoundaryNorm(list(range(cfg.n_levels + 1)), cfg.n_levels)
        ax.imshow(data, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
    else:
        ax.imshow(data, cmap=cfg.cmap, vmin=vmin, vmax=vmax,
                  origin="lower", interpolation="bilinear")

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def _render_colorbar(layer: str, cfg: LayerCfg, out_dir: Path) -> None:
    plt = _plt()
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(1.2, 4), dpi=120)
    fig.subplots_adjust(right=0.4)

    if cfg.n_levels:
        cmap = mcolors.ListedColormap(_ERI_COLORS)
        norm = mcolors.BoundaryNorm(list(range(cfg.n_levels + 1)), cfg.n_levels)
        sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
        cb   = fig.colorbar(sm, cax=ax)
        cb.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cb.set_ticklabels(["0 (safe)", "1", "2", "3", "4 (crisis)"])
    else:
        vmin = cfg.vmin if cfg.vmin is not None else 0.0
        vmax = cfg.vmax if cfg.vmax is not None else 1.0
        sm   = cm.ScalarMappable(cmap=cfg.cmap,
                                  norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, cax=ax)

    ax.set_title(layer, fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{layer}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def render_run_overlays(
    run_dir: Path,
    *,
    recon:    np.ndarray,  # (H, W)           — last-step reconstruction
    forecast: np.ndarray,  # (5, H, W)
    bloom:    np.ndarray,  # (5, H, W) probs
    eri:      np.ndarray,  # (5, H, W) int class
    impact:   np.ndarray,  # (H, W)
    land_mask: np.ndarray, # (H, W)
    bbox: list[float],     # [minlon, minlat, maxlon, maxlat]
    horizons: list[str],   # 5 ISO date strings
    checkpoint_hash: str   = "",
    pipeline_hash: str     = "",
) -> dict:
    """
    Write overlays/<layer>_d{1..5}.png and overlays/colorbars/<layer>.png.
    Returns the metadata dict (also written to run_dir/metadata.json).
    """
    overlay_dir = run_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cbar_dir    = overlay_dir / "colorbars"

    layers_written: dict[str, list[str]] = {k: [] for k in LAYER_CFG}

    for h_idx in range(5):
        label = f"d{h_idx + 1}"
        h_arr = {
            "recon":    recon,           # same for all horizons (last-step)
            "forecast": forecast[h_idx],
            "bloom":    bloom[h_idx],
            "eri":      eri[h_idx].astype(float),
            "impact":   impact,          # same for all horizons
        }
        for layer, arr in h_arr.items():
            out = overlay_dir / f"{layer}_{label}.png"
            _render_layer(arr, land_mask, LAYER_CFG[layer], out)
            layers_written[layer].append(str(out.name))
            log.debug(f"  rendered {out.name}")

    for layer, cfg in LAYER_CFG.items():
        _render_colorbar(layer, cfg, cbar_dir)

    metadata = {
        "bbox":             bbox,
        "crs":              "EPSG:4326",
        "horizons":         horizons,
        "checkpoint_hash":  checkpoint_hash,
        "pipeline_hash":    pipeline_hash,
        "layers":           layers_written,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Overlays written to {overlay_dir}  ({5 * len(LAYER_CFG)} PNGs)")
    return metadata
