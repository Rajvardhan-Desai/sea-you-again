"""
architecture_diagram.py — Generate MM-MARAS v3.4 architecture diagram

Usage:
    python scripts/architecture_diagram.py
    # Output: figures/architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colour palette ───────────────────────────────────────────────────
C_INPUT   = "#4A90D9"   # blue
C_ENCODER = "#27AE60"   # green
C_FUSION  = "#E67E22"   # orange
C_TEMPORAL = "#C0392B"  # red
C_DECODER = "#8E44AD"   # purple
C_HEAD    = "#16A085"   # teal
C_SKIP    = "#7F8C8D"   # grey (skip connections)
C_TEXT    = "#2C3E50"   # dark text
C_BG      = "#FAFAFA"   # background
C_SHAPE   = "#95A5A6"   # shape annotations


def _box(ax, x, y, w, h, label, sublabel, color, fontsize=9, sublabel_size=7):
    """Draw a rounded box with label + sublabel."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=0.90,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h * 0.62, label,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold", color="white", zorder=4,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h * 0.28, sublabel,
            ha="center", va="center", fontsize=sublabel_size,
            color="white", alpha=0.90, zorder=4,
        )


def _arrow(ax, x0, y0, x1, y1, color="#555555", style="-|>", lw=1.2,
           connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=12,
        connectionstyle=connectionstyle,
        zorder=2,
    )
    ax.add_patch(arrow)


def _shape_label(ax, x, y, text, fontsize=6.5):
    """Annotate a tensor shape."""
    ax.text(
        x, y, text,
        ha="center", va="center", fontsize=fontsize,
        color=C_SHAPE, fontstyle="italic", zorder=5,
    )


def create_diagram(save_path: str = "figures/architecture.png"):
    fig, ax = plt.subplots(figsize=(16, 24))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 24)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # ── Title ────────────────────────────────────────────────────────
    ax.text(
        8, 23.5, "MM-MARAS v3.4 Architecture",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=C_TEXT,
    )
    ax.text(
        8, 23.05,
        "Multi-Modal Marine Remote-sensing Analysis & Synthesis System  |  ~44.4M params",
        ha="center", va="center", fontsize=9, color=C_SHAPE,
    )

    # ==================================================================
    # ROW 1: Inputs  (y = 21.0 .. 22.2)
    # ==================================================================
    row_y = 21.2
    bw, bh = 2.6, 1.1
    gap = 0.35
    total_w = 5 * bw + 4 * gap
    start_x = (16 - total_w) / 2

    inputs = [
        ("Optical\n(Chl-a + obs_mask)", "(B,T,2,H,W)"),
        ("Physics\n(SST,SSH,MLD,...+Wind)", "(B,T,12,H,W)"),
        ("Masks\n(obs,MCAR,MNAR,bloom)", "(B,T,4,H,W)"),
        ("BGC Auxiliary\n(O2,NO3,PO4,Si,NPP)", "(B,T,5,H,W)"),
        ("Discharge\n(River + Runoff)", "(B,T,2,H,W)"),
    ]

    input_centers = []
    for i, (label, shape) in enumerate(inputs):
        x = start_x + i * (bw + gap)
        _box(ax, x, row_y, bw, bh, label, shape, C_INPUT, fontsize=8, sublabel_size=6.5)
        input_centers.append(x + bw / 2)

    # ==================================================================
    # ROW 2: Encoders  (y = 18.6 .. 19.7)
    # ==================================================================
    enc_y = 18.6
    enc_h = 1.1

    encoders = [
        ("Swin-UNet\nOpticalEncoder", "D=256"),
        ("Swin-UNet\nPhysicsEncoder", "D=256"),
        ("MaskNet\n(Embed+GNN+Mixer)", "D=256"),
        ("Swin-UNet\nBGCAuxEncoder", "D=256"),
        ("Swin-UNet\nDischargeEncoder", "D=256"),
    ]

    enc_centers = []
    for i, (label, shape) in enumerate(encoders):
        x = start_x + i * (bw + gap)
        _box(ax, x, enc_y, bw, enc_h, label, shape, C_ENCODER, fontsize=8, sublabel_size=7)
        enc_centers.append(x + bw / 2)
        # Arrow from input to encoder
        _arrow(ax, input_centers[i], row_y, enc_centers[i], enc_y + enc_h)

    # Shape annotation below encoders
    _shape_label(ax, 8, 18.3, "Each encoder output: (B, T, 256, 64, 64)")

    # ==================================================================
    # ROW 3: Fusion  (y = 16.4 .. 17.6)
    # ==================================================================
    fus_y = 16.5
    fus_h = 1.2
    fus_w = total_w
    fus_x = start_x

    _box(ax, fus_x, fus_y, fus_w, fus_h,
         "Perceiver IO Cross-Modal Fusion",
         "64 learned latents  |  Cross-Attn (5 streams) + Self-Attn + Decode  |  FP32 attention",
         C_FUSION, fontsize=11, sublabel_size=8)

    fus_cx = fus_x + fus_w / 2

    # Arrows from each encoder to fusion
    for cx in enc_centers:
        _arrow(ax, cx, enc_y, cx, fus_y + fus_h)

    _shape_label(ax, 8, 16.2, "Output: (B, T, 256, 64, 64)")

    # ==================================================================
    # ROW 4: Temporal  (y = 13.8 .. 15.4)
    # ==================================================================
    temp_y = 14.0
    temp_h = 1.3

    # ConvLSTM block
    lstm_w = 5.5
    lstm_x = 2.0
    _box(ax, lstm_x, temp_y, lstm_w, temp_h,
         "2-Layer ConvLSTM",
         "Layer1 → GroupNorm → Layer2 + seq_mean residual  |  cell clamp(+-5)",
         C_TEMPORAL, fontsize=10, sublabel_size=7.5)

    # Temporal Attention block
    attn_w = 5.5
    attn_x = 8.5
    _box(ax, attn_x, temp_y, attn_w, temp_h,
         "Temporal Recon Attention",
         "Cross-attn: final state queries full T-step sequence  |  cosine similarity",
         C_TEMPORAL, fontsize=10, sublabel_size=7.5)

    # Arrow from fusion to ConvLSTM
    _arrow(ax, fus_cx, fus_y, lstm_x + lstm_w / 2, temp_y + temp_h)

    # Arrow from ConvLSTM to Temporal Attention
    _arrow(ax, lstm_x + lstm_w, temp_y + temp_h * 0.5,
           attn_x, temp_y + temp_h * 0.5, lw=1.5)

    # Labels on the arrow
    _shape_label(ax, lstm_x + lstm_w / 2, 13.7, "state (B,256,64,64)\nsequence (B,T,256,64,64)")
    _shape_label(ax, attn_x + attn_w / 2, 13.7, "enriched state (B,256,64,64)")

    # ==================================================================
    # ROW 5: MoE Decoder  (y = 11.2 .. 12.7)
    # ==================================================================
    moe_y = 11.2
    moe_h = 1.6
    moe_w = total_w
    moe_x = start_x

    _box(ax, moe_x, moe_y, moe_w, moe_h,
         "Mixture-of-Experts Decoder (Soft Routing)",
         "",
         C_DECODER, fontsize=11, sublabel_size=8)

    # Sub-boxes inside MoE
    router_w = 2.8
    router_x = moe_x + 0.3
    router_y = moe_y + 0.2
    router_h = 1.2
    box_r = FancyBboxPatch(
        (router_x, router_y), router_w, router_h,
        boxstyle="round,pad=0.02",
        facecolor="#6C3483", edgecolor="white",
        linewidth=1, alpha=0.85, zorder=4,
    )
    ax.add_patch(box_r)
    ax.text(router_x + router_w / 2, router_y + router_h * 0.65,
            "Router", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=5)
    ax.text(router_x + router_w / 2, router_y + router_h * 0.30,
            "GAP → Linear → Softmax\n→ (B, 4) weights", ha="center", va="center",
            fontsize=7, color="white", alpha=0.9, zorder=5)

    # 4 Expert blocks
    exp_w = 2.0
    exp_gap = 0.3
    exp_start = router_x + router_w + 0.6
    expert_labels = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
    for i, elabel in enumerate(expert_labels):
        ex = exp_start + i * (exp_w + exp_gap)
        ey = moe_y + 0.2
        eh = 1.2
        box_e = FancyBboxPatch(
            (ex, ey), exp_w, eh,
            boxstyle="round,pad=0.02",
            facecolor="#6C3483", edgecolor="white",
            linewidth=1, alpha=0.85, zorder=4,
        )
        ax.add_patch(box_e)
        ax.text(ex + exp_w / 2, ey + eh * 0.65, elabel,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=5)
        ax.text(ex + exp_w / 2, ey + eh * 0.30,
                "Conv3x3 → GN\n→ GELU → Conv3x3",
                ha="center", va="center", fontsize=6.5,
                color="white", alpha=0.9, zorder=5)

    # Arrow from temporal attention to MoE
    _arrow(ax, attn_x + attn_w / 2, temp_y,
           moe_x + moe_w / 2, moe_y + moe_h)

    _shape_label(ax, 8, 10.9, "decoded: (B, 256, 64, 64)  |  routing_weights: (B, 4)")

    # ==================================================================
    # ROW 6: Output Heads  (y = 8.2 .. 9.6)
    # ==================================================================
    head_y = 8.4
    head_h = 1.6

    heads = [
        ("ReconHead", "Mask-aware\ndilated convs\n+ optical skip", "(B,1,H,W)\nChl-a recon"),
        ("UncertaintyHead", "1x1 Conv\nlog-var clamp\n[-3, 10]", "(B,1,H,W)\nlog-variance"),
        ("ForecastHead", "Parallel trunk\n+ ConvGRU\nrefinement", "(B,5,H,W)\n5-step forecast"),
        ("ERIHead", "Conv3x3 → GN\n→ GELU → Drop\n→ Conv1x1", "(B,5,H,W)\n5-class ERI"),
        ("BloomFcastHead", "Conv trunk\n+ per-step\n1x1 heads", "(B,5,H,W)\nbloom logits"),
    ]

    head_centers = []
    for i, (name, detail, output) in enumerate(heads):
        x = start_x + i * (bw + gap)
        _box(ax, x, head_y, bw, head_h, name, detail, C_HEAD, fontsize=8, sublabel_size=6.5)
        hcx = x + bw / 2
        head_centers.append(hcx)
        # Arrow from MoE to head
        _arrow(ax, moe_x + moe_w / 2, moe_y, hcx, head_y + head_h)
        # Output shape below
        _shape_label(ax, hcx, 8.05, output, fontsize=6)

    # ==================================================================
    # ROW 7: Loss Functions  (y = 6.0 .. 7.2)
    # ==================================================================
    loss_y = 6.2
    loss_h = 1.0

    losses = [
        ("Recon Loss", "Heteroscedastic\nNLL"),
        ("Holdout Loss", "NLL + Laplacian\n+ L1 bias"),
        ("Forecast Loss", "Huber + SSIM"),
        ("ERI Loss", "Focal ordinal\nCross-Entropy"),
        ("Bloom Loss", "Binary CE\n(pos_weight=10)"),
    ]

    for i, (name, detail) in enumerate(losses):
        x = start_x + i * (bw + gap)
        _box(ax, x, loss_y, bw, loss_h, name, detail, "#34495E", fontsize=7.5, sublabel_size=6)
        lcx = x + bw / 2
        _arrow(ax, head_centers[i], head_y, lcx, loss_y + loss_h)

    # Combined loss annotation
    ax.text(
        8, 5.7,
        "Total = w_recon * L_recon + scale * (w_fcast * L_fcast + w_eri * L_eri + w_bloom * L_bloom) "
        "+ w_holdout * sqrt(scale) * L_holdout + w_aux * L_aux",
        ha="center", va="center", fontsize=7.5, color=C_TEXT,
        fontstyle="italic", zorder=5,
    )
    ax.text(
        8, 5.35,
        "Curriculum: scale ramps 0 → 1.0 over 60% of training  |  "
        "AdamW (lr=5e-5, wd=2e-2)  |  Cosine LR + 5-epoch warmup  |  "
        "AMP (FP16, frozen scaler=8192)",
        ha="center", va="center", fontsize=7, color=C_SHAPE, zorder=5,
    )

    # ==================================================================
    # Skip connections (dashed curves)
    # ==================================================================
    # Optical encoder → ReconHead skip
    _arrow(ax, enc_centers[0] - 0.3, enc_y,
           head_centers[0] - 0.3, head_y + head_h,
           color=C_SKIP, style="-|>", lw=1.0,
           connectionstyle="arc3,rad=-0.25")
    ax.text(
        0.7, 13.5, "optical\nskip",
        ha="center", va="center", fontsize=6.5, color=C_SKIP,
        fontstyle="italic", rotation=90,
    )

    # ConvLSTM sequence → Temporal Attention (already shown as horizontal arrow)
    # Add a curved dashed arrow showing sequence path
    _arrow(ax, lstm_x + lstm_w * 0.7, temp_y,
           attn_x + attn_w * 0.3, temp_y,
           color=C_SKIP, style="-|>", lw=1.0,
           connectionstyle="arc3,rad=-0.3")
    ax.text(
        8, 13.1, "h_sequence (B,T,D,H,W)",
        ha="center", va="center", fontsize=6.5, color=C_SKIP,
        fontstyle="italic",
    )

    # ==================================================================
    # Legend
    # ==================================================================
    legend_y = 4.3
    legend_items = [
        (C_INPUT, "Input Modalities"),
        (C_ENCODER, "Encoders"),
        (C_FUSION, "Cross-Modal Fusion"),
        (C_TEMPORAL, "Temporal Processing"),
        (C_DECODER, "MoE Decoder"),
        (C_HEAD, "Task Heads"),
        ("#34495E", "Loss Functions"),
    ]
    legend_x = 2.0
    for i, (color, label) in enumerate(legend_items):
        lx = legend_x + (i % 4) * 3.3
        ly = legend_y - (i // 4) * 0.4
        box = FancyBboxPatch(
            (lx, ly), 0.35, 0.25,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="none", alpha=0.9,
        )
        ax.add_patch(box)
        ax.text(lx + 0.5, ly + 0.12, label,
                ha="left", va="center", fontsize=7.5, color=C_TEXT)

    # ==================================================================
    # Save
    # ==================================================================
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Architecture diagram saved to: {out_path}")


if __name__ == "__main__":
    create_diagram()
