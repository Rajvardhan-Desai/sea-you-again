"""
model.py — MM-MARAS top-level model (v3)

Architectural fixes for identified bottlenecks:

    [FIX A] ReconHead: 1x1 conv → mask-conditioned spatial inpainting head
            - Takes obs_mask as input so it knows WHERE gaps are
            - Uses 3x3 and 5x5 dilated convs to propagate valid→gap
            - Gets skip connection from optical encoder's last-timestep features
            → Fixes: gap RMSE, gap SSIM, gap bias

    [FIX B] Temporal attention for reconstruction
            - ConvLSTM now returns full hidden sequence (B, T, D, H, W)
            - ReconHead attends back to all T timesteps via cross-attention
            - A gap pixel at t=10 can look up what it saw at t=3
            → Fixes: gap quality, forgotten observation recovery

    [FIX C] Autoregressive forecast refinement
            - After initial parallel prediction, a lightweight GRU refines
              each step conditioned on the previous step's prediction
            → Fixes: forecast SSIM non-monotonicity, long-horizon quality

    [FIX D] ERI from learned features (v3.1: removed bloom_count leakage)
            - v3 concatenated bloom_mask.sum(dim=1) — this was circular since
              the ERI target IS bloom_count thresholded into bins
            - v3.1 removes this, forcing honest prediction from learned features
            → Fixes: ERI metric inflation from label leakage

Unchanged: all encoders, Perceiver IO fusion, MoE decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
from pathlib import Path
_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))
if str(_MODEL_DIR / "encoders") not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR / "encoders"))

from masknet import MaskNet
from optical_encoder import OpticalEncoder
from physics_encoder import PhysicsEncoder
from bgc_encoder import BGCAuxEncoder
from discharge_encoder import DischargeEncoder
from fusion import FusionModule
from moe_decoder import MoEDecoder, compute_aux_loss


# ======================================================================
# Config
# ======================================================================

@dataclass
class ModelConfig:
    T: int = 10
    H: int = 64
    W: int = 64
    H_fcast: int = 5

    C_optical: int = 2
    C_physics: int = 6
    C_wind: int = 4
    C_static: int = 2
    C_masks: int = 4
    C_discharge: int = 2
    C_bgc: int = 5

    embed_dim: int = 256
    n_experts: int = 4
    n_eri_levels: int = 5
    holdout_frac: float = 0.30


# ======================================================================
# [FIX A] Mask-aware spatial ReconHead with skip connection
# ======================================================================

class ReconHead(nn.Module):
    """
    Mask-conditioned spatial reconstruction head.

    Instead of a 1x1 conv (pixel-independent), this head:
    1. Concatenates the obs_mask so it knows which pixels are gaps
    2. Uses dilated convolutions for multi-scale spatial context
    3. Receives a skip connection from the optical encoder for fine detail

    v3.1: Added dilation=8 layer, extending the effective receptive field
    from 9x9 to 25x25 pixels. Real cloud gaps span 15-30+ pixels, so
    interior gap pixels need access to valid observations further away.

    Input:
        decoded:  (B, D, H, W)     from MoE decoder
        opt_skip: (B, D, H, W)     optical encoder features at last timestep
        obs_mask: (B, 1, H, W)     1=valid, 0=gap at last timestep
    Output:
        recon:    (B, 1, H, W)     reconstructed Chl-a
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embed_dim

        # Fuse decoded + skip + mask: D + D + 1 → D
        self.fuse = nn.Sequential(
            nn.Conv2d(D * 2 + 1, D, kernel_size=1, bias=False),
            nn.GroupNorm(8, D),
            nn.GELU(),
        )

        # Multi-scale spatial propagation (cumulative RF: 25x25)
        self.spatial = nn.Sequential(
            # Standard 3x3 — immediate neighbors (RF: 3x3)
            nn.Conv2d(D, D, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D),
            nn.GELU(),
            # Dilated 3x3 (dilation=2) — (RF: 7x7)
            nn.Conv2d(D, D // 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, D // 2),
            nn.GELU(),
            # Dilated 3x3 (dilation=4) — (RF: 15x15)
            nn.Conv2d(D // 2, D // 4, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, D // 4),
            nn.GELU(),
            # [v3.1] Dilated 3x3 (dilation=8) — (RF: 25x25)
            # Reaches 12 pixels from gap boundary — covers most real cloud gaps
            nn.Conv2d(D // 4, D // 4, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.GroupNorm(8, D // 4),
            nn.GELU(),
        )

        self.out_proj = nn.Conv2d(D // 4, 1, kernel_size=1)

    def forward(self, decoded: Tensor, opt_skip: Tensor, obs_mask: Tensor) -> Tensor:
        # obs_mask: (B, H, W) → (B, 1, H, W)
        if obs_mask.ndim == 3:
            obs_mask = obs_mask.unsqueeze(1)

        x = self.fuse(torch.cat([decoded, opt_skip, obs_mask], dim=1))
        x = self.spatial(x)
        return self.out_proj(x)     # (B, 1, H, W)


# ======================================================================
# [FIX B] Temporal attention for reconstruction
# ======================================================================

class TemporalReconAttention(nn.Module):
    """
    Cross-attention from the reconstruction query (last timestep state)
    to the full ConvLSTM hidden sequence.

    A gap pixel at t=10 can attend to t=3 where it was observed,
    recovering details the LSTM may have partially forgotten.

    Input:
        query:    (B, D, H, W)     last hidden state
        sequence: (B, T, D, H, W)  full hidden sequence from layer 1
        obs_mask: (B, T, H, W)     1=valid at each timestep
    Output:
        refined:  (B, D, H, W)     query enriched with temporal context
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Learnable temperature for cosine-similarity attention after L2 norm.
        # Standard 1/sqrt(d) scale compresses logits to ~[-0.125, 0.125] after
        # L2 norm, producing near-uniform attention. A learnable scale lets
        # the model discover the right sharpness.
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1, bias=False)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.norm = nn.GroupNorm(8, embed_dim)

    def forward(
        self, query: Tensor, sequence: Tensor, obs_mask: Tensor
    ) -> Tensor:
        # STRICLY DISABLE AUTOCAST to prevent FP16 norm() squaring overflow
        with torch.amp.autocast("cuda", enabled=False):
            orig_dtype = query.dtype
            query = query.float()
            sequence = sequence.float()
            obs_mask = obs_mask.float()

            B, T, D, H, W = sequence.shape
            nH = self.num_heads
            dH = self.head_dim

            # Query from last timestep: (B, D, H, W)
            q = self.q_proj(query)                          
            q = q.reshape(B, nH, dH, H * W).permute(0, 1, 3, 2)  

            # Keys/values from full sequence: pool each timestep spatially
            seq_flat = sequence.reshape(B * T, D, H, W)
            k = self.k_proj(seq_flat).reshape(B, T, nH, dH, H * W)
            v = self.v_proj(seq_flat).reshape(B, T, nH, dH, H * W)

            # Pool K,V spatially weighted by obs_mask (einsum avoids large broadcast intermediates)
            mask_flat = obs_mask.reshape(B, T, H * W)                          # (B, T, HW)
            mask_sum = mask_flat.sum(dim=-1, keepdim=True).clamp(min=1.0)      # (B, T, 1)
            k_pooled = torch.einsum('bthds,bts->bthd', k, mask_flat) / mask_sum.unsqueeze(2)
            v_pooled = torch.einsum('bthds,bts->bthd', v, mask_flat) / mask_sum.unsqueeze(2)
            del k, v, mask_flat

            k_pooled = k_pooled.permute(0, 2, 1, 3)
            v_pooled = v_pooled.permute(0, 2, 1, 3)

            # L2-normalize Q and K safely to prevent zero-vector gradient explosion
            q_norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            q = q / q_norm
            
            k_norm = k_pooled.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            k_pooled = k_pooled / k_norm

            # Cosine-similarity attention with learnable temperature
            # Clamp logit_scale to prevent softmax overflow (exp(80) ~ 5.5e34, safe in FP32)
            attn = (q @ k_pooled.transpose(-2, -1)) * self.logit_scale.clamp(max=80.0)
            attn = attn.softmax(dim=-1)

            out = attn @ v_pooled                       
            out = out.permute(0, 1, 3, 2).reshape(B, D, H, W)
            out = self.out_proj(out)

            result = self.norm(query + out)
            return result.to(orig_dtype)


# ======================================================================
# [FIX C] Autoregressive forecast refinement
# ======================================================================

class ForecastHead(nn.Module):
    """
    Two-stage forecast head:
    1. Parallel prediction: shared trunk + per-step projection (same as v2)
    2. Autoregressive refinement: lightweight ConvGRU where step t+k
       sees the prediction from step t+k-1

    The refinement is cheap: a single ConvGRU cell with D//4 channels,
    unrolled for 5 steps. It fixes inconsistencies between steps without
    adding significant compute.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embed_dim
        self.H_fcast = cfg.H_fcast

        # Stage 1: parallel prediction (shared trunk + per-step output)
        self.trunk = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D),
            nn.GELU(),
            nn.Conv2d(D, D // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D // 2),
            nn.GELU(),
        )
        self.step_heads = nn.ModuleList([
            nn.Conv2d(D // 2, 1, kernel_size=1)
            for _ in range(cfg.H_fcast)
        ])

        # Stage 2: autoregressive refinement
        # ConvGRU cell: takes (prediction, hidden) → updated prediction
        refine_dim = D // 4
        self.pred_embed = nn.Conv2d(1, refine_dim, kernel_size=3, padding=1)
        self.gru_z = nn.Conv2d(refine_dim * 2, refine_dim, 3, padding=1)
        self.gru_r = nn.Conv2d(refine_dim * 2, refine_dim, 3, padding=1)
        self.gru_n = nn.Conv2d(refine_dim * 2, refine_dim, 3, padding=1)
        self.refine_out = nn.Conv2d(refine_dim, 1, kernel_size=1)
        self.h_init = nn.Conv2d(D, refine_dim, 1)  # warm start from decoded state
        self.refine_dim = refine_dim

    def _gru_step(self, x: Tensor, h: Tensor) -> Tensor:
        """One ConvGRU step."""
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.gru_z(xh))
        r = torch.sigmoid(self.gru_r(xh))
        n = torch.tanh(self.gru_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * n

    def forward(self, decoded: Tensor) -> Tensor:
        B, D, H, W = decoded.shape

        # Stage 1: parallel prediction
        shared = self.trunk(decoded)
        parallel_preds = [head(shared) for head in self.step_heads]

        # Stage 2: autoregressive refinement
        h = self.h_init(decoded)                      # warm start from decoded features
        refined = []
        for t in range(self.H_fcast):
            pred_t = parallel_preds[t]                     # (B, 1, H, W)
            x = self.pred_embed(pred_t)                    # (B, refine_dim, H, W)
            h = self._gru_step(x, h)                       # (B, refine_dim, H, W)
            correction = self.refine_out(h)                # (B, 1, H, W)
            correction = correction.clamp(-1.0, 1.0)       # prevent runaway corrections
            refined.append(pred_t + correction)            # residual refinement

        return torch.cat(refined, dim=1)                   # (B, H_fcast, H, W)


# ======================================================================
# Uncertainty head (unchanged)
# ======================================================================

class UncertaintyHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Conv2d(cfg.embed_dim, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x).clamp(min=-4.0, max=10.0)


# ======================================================================
# [FIX D v3.1] ERI head — predict from learned features only
# ======================================================================

class ERIHead(nn.Module):
    """
    ERI classification from decoded spatiotemporal features.

    v3.1: Removed bloom_count input that was causing label leakage.
    The ERI target is derived from bloom_mask.sum(dim=1) thresholded into
    5 bins — passing that same count as input made the task trivially
    circular. The head now predicts ERI purely from learned representations.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embed_dim
        self.head = nn.Sequential(
            nn.Conv2d(D, D // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D // 2),
            nn.GELU(),
            nn.Conv2d(D // 2, cfg.n_eri_levels, kernel_size=1),
        )

    def forward(self, decoded: Tensor) -> Tensor:
        """
        decoded: (B, D, H, W) — MoE decoder output
        """
        return self.head(decoded)   # (B, 5, H, W)


# ======================================================================
# Bloom forecast head (unchanged from v2)
# ======================================================================

class BloomForecastHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embed_dim
        self.trunk = nn.Sequential(
            nn.Conv2d(D, D // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D // 2),
            nn.GELU(),
        )
        self.step_heads = nn.ModuleList([
            nn.Conv2d(D // 2, 1, kernel_size=1)
            for _ in range(cfg.H_fcast)
        ])

    def forward(self, x: Tensor) -> Tensor:
        shared = self.trunk(x)
        steps = [head(shared) for head in self.step_heads]
        return torch.cat(steps, dim=1)


# ======================================================================
# Ecosystem impact scoring (unchanged from v2)
# ======================================================================

def compute_ecosystem_impact(
    bloom_probs: Tensor, forecast: Tensor, uncertainty: Tensor,
    static: Tensor, land_mask: Tensor,
) -> Tensor:
    ocean = 1.0 - land_mask
    bloom_severity = bloom_probs.max(dim=1).values
    intensity = torch.tanh(forecast.clamp(min=0).max(dim=1).values)
    dist_coast = static[:, 1].clamp(0, 1)
    coastal_weight = 1.0 - dist_coast
    uncertainty_flag = torch.sigmoid(uncertainty.squeeze(1))
    impact = (
        0.40 * bloom_severity + 0.25 * intensity +
        0.20 * coastal_weight + 0.15 * uncertainty_flag
    )
    return (impact * ocean).clamp(0, 1)


# ======================================================================
# Modified TemporalModule — returns full sequence for attention
# ======================================================================

class TemporalModuleV3(nn.Module):
    """
    Same two-layer ConvLSTM, but returns BOTH:
    - state:    (B, D, H, W)      final hidden state (for decoder)
    - sequence: (B, T, D, H, W)   full layer-1 sequence (for recon attention)
    """

    def __init__(self, embed_dim: int = 256, kernel_size: int = 3) -> None:
        super().__init__()
        from temporal import ConvLSTMLayer
        self.embed_dim = embed_dim
        self.layer1 = ConvLSTMLayer(embed_dim, embed_dim, kernel_size)
        self.layer2 = ConvLSTMLayer(embed_dim, embed_dim, kernel_size)
        self.norm1 = nn.GroupNorm(8, embed_dim)
        self.norm2 = nn.GroupNorm(8, embed_dim)

    def forward(self, fused: Tensor) -> tuple[Tensor, Tensor]:
        fused = fused.contiguous()
        B, T, D, H, W = fused.shape

        # Layer 1: full sequence
        h1_seq = self.layer1(fused, return_sequence=True)
        h1_seq = self.norm1(
            h1_seq.reshape(B * T, D, H, W)
        ).reshape(B, T, D, H, W)

        # Layer 2: final state
        seq_mean = fused.mean(dim=1)
        h2_input = h1_seq + seq_mean.unsqueeze(1)
        h2 = self.norm2(self.layer2(h2_input))

        state = h2 + seq_mean

        return state, h1_seq  # state for decoder, h1_seq for recon attention


# ======================================================================
# [v3.1] Structured holdout — synthetic cloud gap generation
# ======================================================================

@torch.no_grad()
def _generate_structured_holdout(
    obs_mask: Tensor,
    target_frac: float = 0.30,
    min_radius: int = 3,
    max_radius: int = 10,
    max_gaps: int = 6,
) -> Tensor:
    """
    Generate spatially structured holdout masks that mimic real cloud gaps.

    Instead of scattering random pixels (which trains the model on noise it
    will never see at inference), this generates random elliptical "cloud
    patches" over observed ocean pixels. The model must reconstruct
    contiguous missing regions — matching real satellite cloud shadows.

    Args:
        obs_mask:     (B, T, H, W)  1 = valid observed pixel
        target_frac:  Target fraction of valid pixels to hold out (~0.30)
        min_radius:   Minimum ellipse semi-axis (pixels)
        max_radius:   Maximum ellipse semi-axis (pixels)
        max_gaps:     Maximum number of elliptical gaps per sample

    Returns:
        holdout: (B, T, H, W)  1 = held-out pixel (subset of obs_mask)
    """
    B, T, H, W = obs_mask.shape
    device = obs_mask.device

    # Pre-compute coordinate grids (shared across batch)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)

    holdout = torch.zeros_like(obs_mask)

    for b in range(B):
        n_gaps = torch.randint(2, max_gaps + 1, (1,)).item()

        cy = torch.randint(0, H, (n_gaps,), device=device).float()
        cx = torch.randint(0, W, (n_gaps,), device=device).float()
        ry = torch.randint(min_radius, max_radius + 1, (n_gaps,), device=device).float()
        rx = torch.randint(min_radius, max_radius + 1, (n_gaps,), device=device).float()
        theta = torch.rand(n_gaps, device=device) * 3.14159

        # Build composite gap mask (H, W) — vectorized over gaps
        # Expand grids: (n_gaps, H, W)
        dy = yy.unsqueeze(0) - cy.view(-1, 1, 1)   # (G, H, W)
        dx = xx.unsqueeze(0) - cx.view(-1, 1, 1)   # (G, H, W)
        c = theta.cos().view(-1, 1, 1)
        s = theta.sin().view(-1, 1, 1)
        u = (dy * c + dx * s) / ry.view(-1, 1, 1)
        v = (-dy * s + dx * c) / rx.view(-1, 1, 1)
        ellipses = (u.pow(2) + v.pow(2)) < 1.0     # (G, H, W)
        gap_mask = ellipses.any(dim=0).float()      # (H, W)

        # Apply to all timesteps, only where pixels are observed
        obs_b = (obs_mask[b] > 0.5).float()         # (T, H, W)
        sample_holdout = gap_mask.unsqueeze(0) * obs_b  # (T, H, W)

        # Hard cap: if holdout exceeds target_frac, randomly drop whole
        # ellipses until we're under budget. This prevents pathological
        # batches with 50-70% holdout that cause gradient explosion.
        valid_count = obs_b.sum()
        if valid_count > 0:
            held_count = sample_holdout.sum()
            actual_frac = held_count / valid_count
            if actual_frac > target_frac * 1.2:
                # Drop ellipses one at a time (largest first via random perm)
                perm = torch.randperm(n_gaps, device=device)
                trimmed = ellipses.clone()
                for idx in perm:
                    trimmed[idx] = False
                    new_mask = trimmed.any(dim=0).float()
                    new_holdout = new_mask.unsqueeze(0) * obs_b
                    new_frac = new_holdout.sum() / valid_count
                    if new_frac <= target_frac:
                        sample_holdout = new_holdout
                        break
                else:
                    sample_holdout = trimmed.any(dim=0).float().unsqueeze(0) * obs_b

        holdout[b] = sample_holdout

    return holdout


# ======================================================================
# Top-level model
# ======================================================================

class MARASSModel(nn.Module):
    """
    MM-MARAS v3.

    Key changes from v2:
    - ReconHead: mask-conditioned spatial head with skip + dilated convs
    - Temporal: returns full sequence; recon uses temporal cross-attention
    - ForecastHead: parallel + autoregressive GRU refinement
    - ERIHead: predicts from learned features only (v3.1: no bloom_count)
    - Holdout: structured cloud gap simulation (v3.1: no random scatter)
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        # Encoders (unchanged)
        self.masknet       = MaskNet(cfg.embed_dim, cfg.T)
        self.opt_enc       = OpticalEncoder(in_channels=cfg.C_optical, embed_dim=cfg.embed_dim)
        self.phy_enc       = PhysicsEncoder(
            C_physics=cfg.C_physics, C_wind=cfg.C_wind,
            C_static=cfg.C_static, embed_dim=cfg.embed_dim,
        )
        self.bgc_enc       = BGCAuxEncoder(C_bgc=cfg.C_bgc, embed_dim=cfg.embed_dim)
        self.discharge_enc = DischargeEncoder(C_discharge=cfg.C_discharge, embed_dim=cfg.embed_dim)

        # Fusion (unchanged)
        self.fusion = FusionModule(embed_dim=cfg.embed_dim, H=cfg.H, W=cfg.W)

        # Temporal (v3: returns full sequence)
        self.temporal = TemporalModuleV3(embed_dim=cfg.embed_dim)

        # Temporal attention for reconstruction [FIX B]
        self.temporal_attn = TemporalReconAttention(cfg.embed_dim, num_heads=4)

        # MoE decoder (unchanged)
        self.decoder = MoEDecoder(embed_dim=cfg.embed_dim, n_experts=cfg.n_experts)

        # Heads
        self.recon_head       = ReconHead(cfg)                # [FIX A]
        self.forecast_head    = ForecastHead(cfg)             # [FIX C]
        self.uncertainty_head = UncertaintyHead(cfg)
        self.eri_head         = ERIHead(cfg)                  # [FIX D]
        self.bloom_fcast_head = BloomForecastHead(cfg)

    def forward(self, batch: dict) -> dict[str, Tensor]:
        cfg = self.cfg

        chl_obs    = batch["chl_obs"]
        obs_mask   = batch["obs_mask"]
        mcar_mask  = batch["mcar_mask"]
        mnar_mask  = batch["mnar_mask"]
        bloom_mask = batch["bloom_mask"]
        physics    = batch["physics"]
        wind       = batch["wind"]
        static     = batch["static"]
        discharge  = batch["discharge"]
        bgc_aux    = batch["bgc_aux"]

        B, T, H, W = chl_obs.shape

        # --- Prepare inputs ---
        optical = torch.stack([chl_obs, obs_mask], dim=2)
        masks   = torch.stack([obs_mask, mcar_mask, mnar_mask, bloom_mask], dim=2)

        holdout_mask = None
        if self.training and cfg.holdout_frac > 0:
            seq_holdout_mask = _generate_structured_holdout(
                obs_mask, target_frac=cfg.holdout_frac,
            )
            optical = optical.clone()
            keep_mask = 1.0 - seq_holdout_mask
            optical[:, :, 0] = optical[:, :, 0] * keep_mask
            optical[:, :, 1] = optical[:, :, 1] * keep_mask
            holdout_mask = seq_holdout_mask[:, -1]

        # --- Encode ---
        mask_emb = self.masknet(masks)
        opt_feat = self.opt_enc(optical)          # (B, T, D, H, W)
        phy_feat = self.phy_enc(physics, wind, static)
        bgc_feat = self.bgc_enc(bgc_aux)
        dis_feat = self.discharge_enc(discharge)

        # --- Fuse ---
        fused = self.fusion(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)

        # --- Temporal (v3: returns state + full sequence) ---
        state, h_sequence = self.temporal(fused)   # (B, D, H, W), (B, T, D, H, W)

        # [FIX B] Temporal attention: enrich final state with sequence
        state_enriched = self.temporal_attn(state, h_sequence, obs_mask)

        # --- Decode ---
        if self.training:
            decoded, routing_weights = self.decoder(state_enriched, return_routing=True)
        else:
            decoded = self.decoder(state_enriched)
            routing_weights = None

        # --- Heads ---

        # [FIX A] ReconHead: mask-aware with optical skip
        obs_mask_last = obs_mask[:, -1]
        if holdout_mask is not None:
            # During training, the obs_mask was zeroed at holdout positions
            # Pass the modified mask so the head knows about holdouts too
            obs_mask_last = obs_mask_last * (1.0 - holdout_mask)
        # [v3] Gate skip by obs_mask — prevent bad encoder features at gap
        # pixels from leaking through the skip connection
        opt_skip = opt_feat[:, -1] * obs_mask_last.unsqueeze(1)
        recon = self.recon_head(decoded, opt_skip, obs_mask_last)

        # [FIX D v3.1] ERI from learned features only (no bloom_count leakage)
        eri = self.eri_head(decoded)

        outputs = {
            "recon":          recon,
            "forecast":       self.forecast_head(decoded),
            "uncertainty":    self.uncertainty_head(decoded),
            "eri":            eri,
            "bloom_forecast": self.bloom_fcast_head(decoded),
        }
        if routing_weights is not None:
            outputs["routing_weights"] = routing_weights
        if holdout_mask is not None:
            outputs["holdout_mask"] = holdout_mask

        return outputs

    def param_count(self) -> dict[str, int]:
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg   = ModelConfig()
    model = MARASSModel(cfg)

    B = 2
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

    expected = {
        "recon":          (B, 1, cfg.H, cfg.W),
        "forecast":       (B, cfg.H_fcast, cfg.H, cfg.W),
        "uncertainty":    (B, 1, cfg.H, cfg.W),
        "eri":            (B, cfg.n_eri_levels, cfg.H, cfg.W),
        "bloom_forecast": (B, cfg.H_fcast, cfg.H, cfg.W),
    }

    # Eval
    model.eval()
    with torch.no_grad():
        out = model(batch)
    print("\n--- Eval shapes ---")
    ok = True
    for k, v in out.items():
        exp = expected.get(k)
        s = "OK" if exp is None or tuple(v.shape) == exp else "MISMATCH"
        print(f"  {k:<18} {str(tuple(v.shape)):<30} {s}")
        if s == "MISMATCH":
            ok = False

    # Train
    model.train()
    out_t = model(batch)
    print("\n--- Train shapes ---")
    for k, v in out_t.items():
        exp = expected.get(k)
        s = "OK" if exp is None or tuple(v.shape) == exp else "MISMATCH"
        print(f"  {k:<18} {str(tuple(v.shape)):<30} {s}")
        if s == "MISMATCH":
            ok = False

    assert "routing_weights" in out_t
    assert "holdout_mask" in out_t

    # Ecosystem impact
    bloom_probs = torch.sigmoid(out["bloom_forecast"])
    impact = compute_ecosystem_impact(
        bloom_probs, out["forecast"], out["uncertainty"],
        batch["static"], torch.zeros(B, cfg.H, cfg.W),
    )
    print(f"\n  impact: {tuple(impact.shape)}  [{impact.min():.3f}, {impact.max():.3f}]")

    # Param count
    print("\n--- Parameters ---")
    for n, c in model.param_count().items():
        print(f"  {n:<22} {c:>10,}")

    if ok:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Shape mismatch.")


if __name__ == "__main__":
    run_smoke_test()