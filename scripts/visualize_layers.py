"""
visualize_layers.py — Layer-wise input/output visualization for MM-MARAS

Registers forward hooks on every major module, runs a single forward pass,
and saves per-module PNG images showing input and output feature maps.

For multi-channel (256-dim) feature maps, the top-K most activated channels
are shown as heatmap grids.  Raw inputs (chl, masks, physics, etc.) are
shown with domain-appropriate colormaps.

Usage:
    # With checkpoint + real patch
    python scripts/visualize_layers.py \\
        --ckpt checkpoints/best.pt \\
        --patch data-preprocessing-pipeline/data/patches/test/0001.npz \\
        --out-dir layer_viz

    # Synthetic data (no files needed)
    python scripts/visualize_layers.py --synthetic --out-dir layer_viz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

# ── repo imports ────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "model" / "encoders"))
sys.path.insert(0, str(_REPO / "model"))
sys.path.insert(0, str(_REPO / "data-preprocessing-pipeline"))

from model import MARASSModel, ModelConfig  # noqa: E402

# ── constants ───────────────────────────────────────────────────────────
TOP_K_DEFAULT = 16
PANEL_SIZE = 1.4  # inches per subplot

PHYSICS_NAMES = ["thetao", "uo", "vo", "mlotst", "zos", "so"]
WIND_NAMES = ["u10", "v10", "msl", "tp"]
STATIC_NAMES = ["bathymetry", "dist_coast"]
DISCHARGE_NAMES = ["dis24", "rowe"]
BGC_NAMES = ["o2", "no3", "po4", "si", "nppv"]
MASK_NAMES = ["obs_mask", "mcar_mask", "mnar_mask", "bloom_mask"]
OPTICAL_NAMES = ["chl_obs", "obs_mask"]

SILENT_NAN_KEYS = {"static", "physics", "discharge", "bgc_aux", "target_chl"}


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize layer inputs/outputs")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint .pt path")
    p.add_argument("--patch", type=str, default=None, help="Single .npz patch path")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    p.add_argument("--out-dir", default="layer_viz", help="Output directory")
    p.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help="Channels to show")
    p.add_argument("--device", default=None)
    p.add_argument("--timestep", type=int, default=-1, help="Timestep to visualize")
    return p.parse_args()


# =====================================================================
# Data loading
# =====================================================================

def make_synthetic_batch(device: torch.device) -> dict[str, Tensor]:
    cfg = ModelConfig()
    B = 1
    batch = {
        "chl_obs":    torch.randn(B, cfg.T, cfg.H, cfg.W),
        "obs_mask":   torch.randint(0, 2, (B, cfg.T, cfg.H, cfg.W)).float(),
        "mcar_mask":  torch.zeros(B, cfg.T, cfg.H, cfg.W),
        "mnar_mask":  torch.zeros(B, cfg.T, cfg.H, cfg.W),
        "bloom_mask": (torch.rand(B, cfg.T, cfg.H, cfg.W) > 0.95).float(),
        "physics":    torch.randn(B, cfg.T, cfg.C_physics, cfg.H, cfg.W),
        "wind":       torch.randn(B, cfg.T, cfg.C_wind, cfg.H, cfg.W),
        "static":     torch.randn(B, cfg.C_static, cfg.H, cfg.W),
        "discharge":  torch.randn(B, cfg.T, cfg.C_discharge, cfg.H, cfg.W),
        "bgc_aux":    torch.randn(B, cfg.T, cfg.C_bgc, cfg.H, cfg.W),
    }
    return {k: v.to(device) for k, v in batch.items()}


def load_patch(path: str, device: torch.device) -> dict[str, Tensor]:
    data = np.load(path, allow_pickle=False)
    sample: dict[str, Tensor] = {}

    static_raw = data["static"].astype(np.float32)
    land_mask = np.isnan(static_raw).any(axis=0).astype(np.float32)
    sample["land_mask"] = torch.from_numpy(land_mask)

    for key in data.keys():
        arr = data[key].astype(np.float32)
        t = torch.from_numpy(arr)
        if key == "chl_obs":
            t = torch.nan_to_num(t, nan=0.0)
        elif key in SILENT_NAN_KEYS:
            t = torch.nan_to_num(t, nan=0.0)
        sample[key] = t

    # add batch dim
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    return batch


# =====================================================================
# Hook infrastructure
# =====================================================================

class HookStore:
    def __init__(self) -> None:
        self.inputs: dict[str, tuple] = {}
        self.outputs: dict[str, object] = {}
        self._handles: list = []

    def register(self, name: str, module: nn.Module) -> None:
        def hook_fn(mod: nn.Module, inp: tuple, out: object, _name: str = name) -> None:
            self.inputs[_name] = tuple(
                x.detach().cpu() if isinstance(x, Tensor) else x for x in inp
            )
            if isinstance(out, tuple):
                self.outputs[_name] = tuple(
                    x.detach().cpu() if isinstance(x, Tensor) else x for x in out
                )
            else:
                self.outputs[_name] = out.detach().cpu() if isinstance(out, Tensor) else out

        self._handles.append(module.register_forward_hook(hook_fn))

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def register_all_hooks(store: HookStore, model: MARASSModel) -> None:
    targets = [
        ("masknet",          model.masknet),
        ("opt_enc",          model.opt_enc),
        ("phy_enc",          model.phy_enc),
        ("bgc_enc",          model.bgc_enc),
        ("discharge_enc",    model.discharge_enc),
        ("fusion",           model.fusion),
        ("temporal",         model.temporal),
        ("temporal_attn",    model.temporal_attn),
        ("decoder",          model.decoder),
        ("recon_head",       model.recon_head),
        ("forecast_head",    model.forecast_head),
        ("uncertainty_head", model.uncertainty_head),
        ("eri_head",         model.eri_head),
        ("bloom_fcast_head", model.bloom_fcast_head),
    ]
    for name, module in targets:
        store.register(name, module)


# =====================================================================
# Visualization helpers
# =====================================================================

def select_top_k(feat: Tensor, k: int) -> tuple[list[int], Tensor]:
    """Select top-k channels by mean absolute activation from (C, H, W)."""
    C = feat.shape[0]
    k = min(k, C)
    activation = feat.abs().mean(dim=(1, 2))
    idx = activation.topk(k).indices.sort().values
    return idx.tolist(), feat[idx]


def _grid_layout(n: int) -> tuple[int, int]:
    """Compute (nrows, ncols) for n panels, preferring wider grids."""
    if n <= 4:
        return 1, n
    if n <= 8:
        return 2, (n + 1) // 2
    if n <= 16:
        return 2, (n + 1) // 2
    return 4, (n + 3) // 4


def plot_channel_grid(
    arrays: list[np.ndarray],
    titles: list[str],
    suptitle: str,
    save_path: Path,
    cmap: str = "viridis",
    cmaps: list[str] | None = None,
) -> None:
    """Save a grid of 2D heatmaps."""
    n = len(arrays)
    if n == 0:
        return
    nrows, ncols = _grid_layout(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * PANEL_SIZE, nrows * PANEL_SIZE + 0.6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            cm = cmaps[i] if cmaps else cmap
            im = ax.imshow(arrays[i], cmap=cm, aspect="equal")
            ax.set_title(titles[i], fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")

    fig.suptitle(suptitle, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_io_figure(
    in_arrays: list[np.ndarray],
    in_titles: list[str],
    out_arrays: list[np.ndarray],
    out_titles: list[str],
    suptitle: str,
    save_path: Path,
    in_cmap: str = "viridis",
    out_cmap: str = "viridis",
    in_cmaps: list[str] | None = None,
    out_cmaps: list[str] | None = None,
) -> None:
    """Save a two-row figure: top=input, bottom=output."""
    n_in = len(in_arrays)
    n_out = len(out_arrays)
    ncols = max(n_in, n_out, 1)
    fig, axes = plt.subplots(2, ncols, figsize=(ncols * PANEL_SIZE, 2 * PANEL_SIZE + 0.8))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    for col in range(ncols):
        # input row
        ax = axes[0, col]
        if col < n_in:
            cm = in_cmaps[col] if in_cmaps else in_cmap
            im = ax.imshow(in_arrays[col], cmap=cm, aspect="equal")
            ax.set_title(in_titles[col], fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

        # output row
        ax = axes[1, col]
        if col < n_out:
            cm = out_cmaps[col] if out_cmaps else out_cmap
            im = ax.imshow(out_arrays[col], cmap=cm, aspect="equal")
            ax.set_title(out_titles[col], fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("Input", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Output", fontsize=9, fontweight="bold")
    fig.suptitle(suptitle, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _to_np(t: Tensor) -> np.ndarray:
    return t.float().numpy()


def _feat_to_topk(feat: Tensor, k: int) -> tuple[list[np.ndarray], list[str]]:
    """From (C, H, W), return top-k arrays and titles."""
    idx, slices = select_top_k(feat, k)
    arrays = [_to_np(slices[i]) for i in range(slices.shape[0])]
    titles = [f"ch {idx[i]}" for i in range(len(idx))]
    return arrays, titles


# =====================================================================
# Per-module visualization functions
# =====================================================================

def visualize_raw_inputs(batch: dict[str, Tensor], out_dir: Path, t: int) -> None:
    """01_raw_inputs.png — all raw fields at the chosen timestep."""
    arrays, titles, cmaps = [], [], []

    # chl_obs
    chl = _to_np(batch["chl_obs"][0, t].cpu())
    arrays.append(chl); titles.append("chl_obs"); cmaps.append("YlGn")

    # masks
    for name in ["obs_mask", "mcar_mask", "mnar_mask", "bloom_mask"]:
        arrays.append(_to_np(batch[name][0, t].cpu()))
        titles.append(name)
        cmaps.append("gray_r")

    # physics channels
    velocity_idx = {1, 2}  # uo, vo
    for i, pname in enumerate(PHYSICS_NAMES):
        arrays.append(_to_np(batch["physics"][0, t, i].cpu()))
        titles.append(pname)
        cmaps.append("RdBu_r" if i in velocity_idx else "viridis")

    # wind channels
    wind_vel_idx = {0, 1}  # u10, v10
    for i, wname in enumerate(WIND_NAMES):
        arrays.append(_to_np(batch["wind"][0, t, i].cpu()))
        titles.append(wname)
        cmaps.append("RdBu_r" if i in wind_vel_idx else "viridis")

    # static (not time-indexed)
    for i, sname in enumerate(STATIC_NAMES):
        arrays.append(_to_np(batch["static"][0, i].cpu()))
        titles.append(sname)
        cmaps.append("terrain" if i == 0 else "viridis")

    # discharge
    for i, dname in enumerate(DISCHARGE_NAMES):
        arrays.append(_to_np(batch["discharge"][0, t, i].cpu()))
        titles.append(dname)
        cmaps.append("Blues")

    # bgc_aux
    for i, bname in enumerate(BGC_NAMES):
        arrays.append(_to_np(batch["bgc_aux"][0, t, i].cpu()))
        titles.append(bname)
        cmaps.append("viridis")

    plot_channel_grid(arrays, titles, f"Raw Inputs (t={t})", out_dir / "01_raw_inputs.png",
                      cmaps=cmaps)


def visualize_chl_temporal(batch: dict[str, Tensor], out_dir: Path) -> None:
    """02_chl_temporal.png — chl_obs across all timesteps."""
    T = batch["chl_obs"].shape[1]
    arrays = [_to_np(batch["chl_obs"][0, t_i].cpu()) for t_i in range(T)]
    titles = [f"t={t_i}" for t_i in range(T)]
    plot_channel_grid(arrays, titles, "Chl-a Temporal Sequence", out_dir / "02_chl_temporal.png",
                      cmap="YlGn")


def _visualize_encoder(
    store: HookStore,
    hook_name: str,
    in_names: list[str],
    label: str,
    num: str,
    out_dir: Path,
    top_k: int,
    t: int,
    in_cmap: str = "viridis",
    in_cmaps: list[str] | None = None,
) -> None:
    """Generic encoder visualization: input channels -> top-K output channels."""
    inp = store.inputs[hook_name]
    out = store.outputs[hook_name]

    # Collect input arrays at the target timestep
    in_arrays, in_titles = [], []
    if hook_name == "phy_enc":
        # 3 separate tensors: physics (B,T,6,H,W), wind (B,T,4,H,W), static (B,2,H,W)
        physics, wind, static = inp[0], inp[1], inp[2]
        phy_cmaps = []
        for i, pn in enumerate(PHYSICS_NAMES):
            in_arrays.append(_to_np(physics[0, t, i]))
            in_titles.append(pn)
            phy_cmaps.append("RdBu_r" if i in (1, 2) else "viridis")
        for i, wn in enumerate(WIND_NAMES):
            in_arrays.append(_to_np(wind[0, t, i]))
            in_titles.append(wn)
            phy_cmaps.append("RdBu_r" if i in (0, 1) else "viridis")
        for i, sn in enumerate(STATIC_NAMES):
            in_arrays.append(_to_np(static[0, i]))
            in_titles.append(sn)
            phy_cmaps.append("viridis")
        in_cmaps = phy_cmaps
    else:
        # Single tensor: (B, T, C, H, W)
        tensor = inp[0]
        if tensor.ndim == 5:
            C = tensor.shape[2]
            for i in range(C):
                in_arrays.append(_to_np(tensor[0, t, i]))
                in_titles.append(in_names[i] if i < len(in_names) else f"ch {i}")
        elif tensor.ndim == 4:
            C = tensor.shape[1]
            for i in range(C):
                in_arrays.append(_to_np(tensor[0, i]))
                in_titles.append(in_names[i] if i < len(in_names) else f"ch {i}")

    # Output: (B, T, D, H, W) -> take [0, t, :, :, :]
    if isinstance(out, tuple):
        out_tensor = out[0]
    else:
        out_tensor = out
    if out_tensor.ndim == 5:
        feat = out_tensor[0, t]  # (D, H, W)
    else:
        feat = out_tensor[0]  # (D, H, W)

    out_arrays, out_titles = _feat_to_topk(feat, top_k)

    plot_io_figure(
        in_arrays, in_titles, out_arrays, out_titles,
        f"{label} — Input / Output (t={t})",
        out_dir / f"{num}_{hook_name}.png",
        in_cmaps=in_cmaps, in_cmap=in_cmap,
    )


def visualize_encoders(store: HookStore, out_dir: Path, top_k: int, t: int) -> None:
    _visualize_encoder(store, "masknet", MASK_NAMES, "MaskNet Encoder",
                       "03", out_dir, top_k, t, in_cmap="gray_r",
                       in_cmaps=["gray_r"] * 4)
    _visualize_encoder(store, "opt_enc", OPTICAL_NAMES, "Optical Encoder",
                       "04", out_dir, top_k, t,
                       in_cmaps=["YlGn", "gray_r"])
    _visualize_encoder(store, "phy_enc", [], "Physics Encoder",
                       "05", out_dir, top_k, t)
    _visualize_encoder(store, "bgc_enc", BGC_NAMES, "BGC Encoder",
                       "06", out_dir, top_k, t)
    _visualize_encoder(store, "discharge_enc", DISCHARGE_NAMES, "Discharge Encoder",
                       "07", out_dir, top_k, t, in_cmap="Blues")


def visualize_fusion(store: HookStore, out_dir: Path, top_k: int, t: int) -> None:
    """08_fusion.png — mean of 5 encoder outputs vs fused output."""
    # Gather the 5 encoder outputs to compute their mean (the "before fusion" baseline)
    enc_names = ["opt_enc", "phy_enc", "masknet", "bgc_enc", "discharge_enc"]
    enc_feats = []
    for en in enc_names:
        o = store.outputs[en]
        o = o[0] if isinstance(o, tuple) else o
        if o.ndim == 5:
            enc_feats.append(o[0, t])  # (D, H, W)
        else:
            enc_feats.append(o[0])
    mean_enc = torch.stack(enc_feats).mean(dim=0)  # (D, H, W)

    in_arrays, in_titles = _feat_to_topk(mean_enc, top_k)

    fused = store.outputs["fusion"]
    fused = fused[0] if isinstance(fused, tuple) else fused
    if fused.ndim == 5:
        fused_feat = fused[0, t]
    else:
        fused_feat = fused[0]
    out_arrays, out_titles = _feat_to_topk(fused_feat, top_k)

    plot_io_figure(
        in_arrays, in_titles, out_arrays, out_titles,
        f"Perceiver IO Fusion — Mean Encoders / Fused (t={t})",
        out_dir / "08_fusion.png",
    )


def visualize_temporal(store: HookStore, out_dir: Path, top_k: int, t: int) -> None:
    """09_temporal.png and 10_temporal_sequence.png."""
    # Input: fused (B, T, D, H, W)
    fused_inp = store.inputs["temporal"][0]
    if fused_inp.ndim == 5:
        in_feat = fused_inp[0, t]
    else:
        in_feat = fused_inp[0]
    in_arrays, in_titles = _feat_to_topk(in_feat, top_k)

    # Output: (state, h_sequence) tuple
    temp_out = store.outputs["temporal"]
    state = temp_out[0][0] if isinstance(temp_out, tuple) else temp_out[0]  # (D, H, W)
    out_arrays, out_titles = _feat_to_topk(state, top_k)

    plot_io_figure(
        in_arrays, in_titles, out_arrays, out_titles,
        "ConvLSTM Temporal — Fused Input / Final State",
        out_dir / "09_temporal.png",
    )

    # 10_temporal_sequence.png — top-4 channels across all timesteps
    if isinstance(temp_out, tuple) and len(temp_out) > 1:
        h_seq = temp_out[1]  # (B, T, D, H, W)
        if h_seq is not None and isinstance(h_seq, Tensor) and h_seq.ndim == 5:
            T_len = h_seq.shape[1]
            # Find top-4 channels from the last timestep
            last_feat = h_seq[0, -1]  # (D, H, W)
            top4_idx, _ = select_top_k(last_feat, 4)

            arrays, titles = [], []
            for ch_i in top4_idx:
                for t_i in range(T_len):
                    arrays.append(_to_np(h_seq[0, t_i, ch_i]))
                    titles.append(f"ch{ch_i} t={t_i}")

            n = len(arrays)
            ncols = T_len
            nrows = len(top4_idx)
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * PANEL_SIZE, nrows * PANEL_SIZE + 0.6))
            if not isinstance(axes, np.ndarray):
                axes = np.array([[axes]])
            if axes.ndim == 1:
                axes = axes.reshape(1, -1)

            for r in range(nrows):
                for c in range(ncols):
                    ax = axes[r, c]
                    idx = r * ncols + c
                    if idx < n:
                        im = ax.imshow(arrays[idx], cmap="viridis", aspect="equal")
                        ax.set_title(titles[idx], fontsize=6)
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.suptitle("ConvLSTM Hidden Sequence — Top-4 Channels x Timesteps",
                         fontsize=10, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(out_dir / "10_temporal_sequence.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def visualize_temporal_attn(store: HookStore, out_dir: Path, top_k: int) -> None:
    """11_temporal_attn.png."""
    inp = store.inputs["temporal_attn"]
    query = inp[0]  # (B, D, H, W)
    in_arrays, in_titles = _feat_to_topk(query[0], top_k)

    out = store.outputs["temporal_attn"]
    out_t = out[0] if isinstance(out, tuple) else out
    out_arrays, out_titles = _feat_to_topk(out_t[0], top_k)

    plot_io_figure(
        in_arrays, in_titles, out_arrays, out_titles,
        "Temporal Recon Attention — Query / Enriched State",
        out_dir / "11_temporal_attn.png",
    )


def visualize_decoder(store: HookStore, out_dir: Path, top_k: int) -> None:
    """12_decoder.png."""
    inp = store.inputs["decoder"]
    in_t = inp[0]  # (B, D, H, W)
    in_arrays, in_titles = _feat_to_topk(in_t[0], top_k)

    out = store.outputs["decoder"]
    out_t = out[0] if isinstance(out, tuple) else out
    out_arrays, out_titles = _feat_to_topk(out_t[0], top_k)

    plot_io_figure(
        in_arrays, in_titles, out_arrays, out_titles,
        "MoE Decoder — Input / Decoded Output",
        out_dir / "12_decoder.png",
    )


def _visualize_head(
    store: HookStore,
    hook_name: str,
    label: str,
    num: str,
    out_dir: Path,
    top_k: int,
    out_names: list[str] | None = None,
    out_cmap: str = "coolwarm",
) -> None:
    """Visualize a task head: decoded input top-K vs small-channel output."""
    inp = store.inputs[hook_name]
    # First input is always decoded (B, D, H, W)
    decoded = inp[0]
    in_arrays, in_titles = _feat_to_topk(decoded[0], top_k)

    out = store.outputs[hook_name]
    out_t = out[0] if isinstance(out, tuple) else out
    C_out = out_t.shape[1]
    out_arrays = [_to_np(out_t[0, c]) for c in range(C_out)]
    if out_names and len(out_names) == C_out:
        out_titles = out_names
    else:
        out_titles = [f"ch {c}" for c in range(C_out)]

    plot_io_figure(
        in_arrays, in_titles, out_arrays, out_titles,
        f"{label} — Decoded Input / Output",
        out_dir / f"{num}_{hook_name}.png",
        out_cmap=out_cmap,
    )


def visualize_heads(store: HookStore, out_dir: Path, top_k: int) -> None:
    _visualize_head(store, "recon_head", "Reconstruction Head", "13", out_dir, top_k,
                    out_names=["recon"], out_cmap="coolwarm")
    _visualize_head(store, "forecast_head", "Forecast Head", "14", out_dir, top_k,
                    out_names=[f"t+{i+1}" for i in range(5)], out_cmap="coolwarm")
    _visualize_head(store, "uncertainty_head", "Uncertainty Head", "15", out_dir, top_k,
                    out_names=["log_var"], out_cmap="magma")
    _visualize_head(store, "eri_head", "ERI Head", "16", out_dir, top_k,
                    out_names=[f"level_{i}" for i in range(5)], out_cmap="YlOrRd")
    _visualize_head(store, "bloom_fcast_head", "Bloom Forecast Head", "17", out_dir, top_k,
                    out_names=[f"bloom_t+{i+1}" for i in range(5)], out_cmap="Reds")


def visualize_pipeline_overview(
    batch: dict[str, Tensor],
    store: HookStore,
    model_outputs: dict[str, Tensor],
    out_dir: Path,
    t: int,
) -> None:
    """00_pipeline_overview.png — one representative from each stage."""
    stages: list[tuple[str, np.ndarray]] = []

    # 1. Raw input: chl_obs
    stages.append(("chl_obs", _to_np(batch["chl_obs"][0, t].cpu())))

    # 2-6. Encoder outputs (top-1 channel each)
    for enc_name, label in [
        ("opt_enc", "Optical Enc"),
        ("phy_enc", "Physics Enc"),
        ("masknet", "MaskNet"),
        ("bgc_enc", "BGC Enc"),
        ("discharge_enc", "Discharge Enc"),
    ]:
        o = store.outputs[enc_name]
        o = o[0] if isinstance(o, tuple) else o
        feat = o[0, t] if o.ndim == 5 else o[0]  # (D, H, W)
        _, top1 = select_top_k(feat, 1)
        stages.append((label, _to_np(top1[0])))

    # 7. Fusion
    fused = store.outputs["fusion"]
    fused = fused[0] if isinstance(fused, tuple) else fused
    f_feat = fused[0, t] if fused.ndim == 5 else fused[0]
    _, top1 = select_top_k(f_feat, 1)
    stages.append(("Fusion", _to_np(top1[0])))

    # 8. Temporal state
    temp_out = store.outputs["temporal"]
    state = temp_out[0] if isinstance(temp_out, tuple) else temp_out
    _, top1 = select_top_k(state[0], 1)
    stages.append(("Temporal", _to_np(top1[0])))

    # 9. Temporal attention
    attn_out = store.outputs["temporal_attn"]
    attn_t = attn_out[0] if isinstance(attn_out, tuple) else attn_out
    _, top1 = select_top_k(attn_t[0], 1)
    stages.append(("Temp Attn", _to_np(top1[0])))

    # 10. Decoder
    dec_out = store.outputs["decoder"]
    dec_t = dec_out[0] if isinstance(dec_out, tuple) else dec_out
    _, top1 = select_top_k(dec_t[0], 1)
    stages.append(("MoE Dec", _to_np(top1[0])))

    # 11-15. Task head outputs
    for key, label in [
        ("recon", "Recon"), ("forecast", "Fcast t+1"),
        ("uncertainty", "Uncert"), ("eri", "ERI"),
        ("bloom_forecast", "Bloom"),
    ]:
        out_t = model_outputs[key].cpu()
        stages.append((label, _to_np(out_t[0, 0])))

    # Plot
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(n * PANEL_SIZE, PANEL_SIZE + 0.7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, (label, arr) in enumerate(stages):
        ax = axes[i]
        im = ax.imshow(arr, cmap="viridis", aspect="equal")
        ax.set_title(label, fontsize=6, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # draw arrows between stages
    fig.suptitle("MM-MARAS Pipeline Overview — One Channel Per Stage",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_dir / "00_pipeline_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()

    if not args.synthetic and not args.ckpt and not args.patch:
        log.info("No --ckpt/--patch provided, using --synthetic mode")
        args.synthetic = True

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log.info(f"Device: {device}")

    # Load model
    cfg = ModelConfig()
    model = MARASSModel(cfg).to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        log.info(f"Loaded checkpoint: {args.ckpt}")
    else:
        log.info("Using randomly initialized model (no checkpoint)")

    model.eval()

    # Load data
    if args.patch:
        batch = load_patch(args.patch, device)
        log.info(f"Loaded patch: {args.patch}")
    else:
        batch = make_synthetic_batch(device)
        log.info("Using synthetic data")

    # Register hooks
    store = HookStore()
    register_all_hooks(store, model)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    log.info("Forward pass complete")

    # Move model outputs to CPU
    outputs = {k: v.detach().cpu() for k, v in outputs.items() if isinstance(v, Tensor)}

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = args.timestep
    top_k = args.top_k

    # Generate all visualizations
    log.info("Generating visualizations...")
    visualize_raw_inputs(batch, out_dir, t)
    visualize_chl_temporal(batch, out_dir)
    visualize_encoders(store, out_dir, top_k, t)
    visualize_fusion(store, out_dir, top_k, t)
    visualize_temporal(store, out_dir, top_k, t)
    visualize_temporal_attn(store, out_dir, top_k)
    visualize_decoder(store, out_dir, top_k)
    visualize_heads(store, out_dir, top_k)
    visualize_pipeline_overview(batch, store, outputs, out_dir, t)

    store.remove_all()

    n_figs = len(list(out_dir.glob("*.png")))
    log.info(f"Saved {n_figs} figures to {out_dir}/")


if __name__ == "__main__":
    main()
