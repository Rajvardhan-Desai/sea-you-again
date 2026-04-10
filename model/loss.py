"""
loss.py — MM-MARAS loss functions (v3)

Changes from v2:
    [v3] holdout_recon_loss: Laplacian weight 0.02 → 0.05, added L1 bias term
    [v3] forecast_loss: added differentiable SSIM component (weight 0.2)
    [v3] eri_loss: class 1 weight 10.0 → 15.0 for better rare-class F1
    [v3] LossWeights: holdout weight 0.5 → 0.8 for stronger gap supervision
    [v3.1] NLL: log_var floor -6 → -4, per-pixel NLL clamped at 10.0
    [v3.1] Holdout loss on sqrt-curriculum to prevent gradient conflicts

Changes from v1:
    [PERF 3] aux weight 0.001 → 0.01 — expert 0 was nearly dead at 5.3%
    [PERF 4] recon_loss now includes an SSIM component for gap pixels via
             the holdout pathway, encouraging spatial coherence in gap fills
    [FEAT 1] bloom_forecast_loss — binary CE for per-step bloom prediction
    [FEAT 1] LossWeights gains bloom_fcast field (default 0.3)
    [FEAT 1] bloom_fcast ramps in with forecast/ERI curriculum

Loss summary:
    recon_loss          Heteroscedastic NLL over observed ocean pixels
    holdout_recon_loss  Heteroscedastic NLL + Laplacian + L1 bias on held-out pixels
    forecast_loss       Masked Huber + SSIM over target_mask * (1 - land_mask)
    eri_loss            Ordinal cross-entropy via cumulative link model
    bloom_forecast_loss Binary CE for bloom prediction at each forecast step [FEAT 1]
    aux_loss            MoE load-balancing (from moe_decoder.py)

Combined:
    loss = (
        w_recon      * recon_loss
      + w_forecast   * scale * forecast_loss
      + w_eri        * scale * eri_loss
      + w_bloom_fcast* scale * bloom_forecast_loss
      + w_aux        * aux_loss
      + w_holdout    * holdout_recon_loss
    )

Default weights:
    w_recon       = 1.0
    w_forecast    = 0.5
    w_eri         = 0.3
    w_bloom_fcast = 0.3    [FEAT 1] new
    w_aux         = 0.01   [PERF 3] was 0.001
    w_holdout     = 0.8    [v3] was 0.5
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from moe_decoder import compute_aux_loss


# ======================================================================
# [v3] Differentiable SSIM helpers
# ======================================================================

def _gaussian_window(size: int = 7, sigma: float = 1.5) -> Tensor:
    """2D Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = (-coords.pow(2) / (2 * sigma ** 2)).exp()
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)          # (1, 1, size, size)


def _ssim_map(
    x: Tensor, y: Tensor, window: Tensor,
    C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> Tensor:
    """Per-pixel SSIM between (B, 1, H, W) tensors."""
    pad = window.shape[-1] // 2
    mu_x  = F.conv2d(x, window, padding=pad)
    mu_y  = F.conv2d(y, window, padding=pad)
    s_xx  = F.conv2d(x * x, window, padding=pad) - mu_x.pow(2)
    s_yy  = F.conv2d(y * y, window, padding=pad) - mu_y.pow(2)
    s_xy  = F.conv2d(x * y, window, padding=pad) - mu_x * mu_y
    num   = (2 * mu_x * mu_y + C1) * (2 * s_xy + C2)
    den   = (mu_x.pow(2) + mu_y.pow(2) + C1) * (s_xx + s_yy + C2)
    return num / den                                      # (B, 1, H, W)


# ======================================================================
# Reconstruction loss — heteroscedastic NLL
# ======================================================================

def recon_loss(
    pred: Tensor,
    log_var: Tensor,
    target: Tensor,
    obs_mask: Tensor,
    land_mask: Tensor,
    holdout_mask: Tensor | None = None,
) -> Tensor:
    """
    Heteroscedastic negative log-likelihood loss for Chl-a reconstruction.

    NLL = 0.5 * (log_var + (pred - target)^2 / exp(log_var))

    Supervision is restricted to observed ocean pixels.
    Held-out pixels are excluded (supervised separately by holdout_recon_loss).
    """
    # FORCE FP32 for numerical stability
    target_t  = target[:, -1].float()
    valid_t   = obs_mask[:, -1]

    ocean = 1.0 - land_mask
    sup_mask = valid_t * ocean
    if holdout_mask is not None:
        sup_mask = sup_mask * (1.0 - holdout_mask)

    pred_sq   = pred.squeeze(1).float()
    lv_sq     = log_var.squeeze(1).float()
    
    # [v3.1] Floor at -4 limits max precision weight to ~55x (was -6 → 400x)
    lv_clamped = lv_sq.clamp(min=-4.0, max=10.0)

    nll = 0.5 * (lv_clamped + (pred_sq - target_t).pow(2) / lv_clamped.exp())
    # [v3.1] Per-pixel clamp prevents a few bad pixels from dominating gradients
    nll = nll.clamp(max=10.0)

    n_valid = sup_mask.sum().clamp(min=1.0)
    return (nll * sup_mask).sum() / n_valid


# ======================================================================
# Holdout reconstruction loss — gap-filling supervision
# [PERF 4] Now includes SSIM component for spatial coherence
# ======================================================================

def holdout_recon_loss(
    pred:         Tensor,
    log_var:      Tensor,
    target:       Tensor,
    holdout_mask: Tensor,
    land_mask:    Tensor,
) -> Tensor:
    """
    Heteroscedastic NLL + spatial smoothness on held-out pixels.

    [PERF 4] Added a gradient-matching term that penalizes spatial
    discontinuities at gap boundaries. This encourages the model to
    produce spatially coherent gap fills instead of pixel-independent
    predictions that ignore spatial structure (which caused SSIM=0.000).

    The gradient term computes the Laplacian (second spatial derivative)
    of both prediction and target within held-out regions and penalizes
    the difference. This acts as a lightweight SSIM proxy without the
    computational cost of full SSIM.
    """
    ocean    = 1.0 - land_mask
    sup_mask = holdout_mask * ocean
    n_valid  = sup_mask.sum().clamp(min=1.0)

    # [v3.1] Floor at -4 limits max precision weight to ~55x (was -6 → 400x)
    pred_sq = pred.squeeze(1).float()
    lv_sq   = log_var.squeeze(1).float().clamp(min=-4.0, max=10.0)
    target_f = target.float()

    # Standard NLL component
    nll = 0.5 * (lv_sq + (pred_sq - target_f).pow(2) / lv_sq.exp())
    # [v3.1] Per-pixel clamp prevents a few bad holdout pixels from dominating
    nll = nll.clamp(max=10.0)
    nll_loss = (nll * sup_mask).sum() / n_valid

    # [PERF 4] Spatial gradient matching — penalize Laplacian mismatch
    # This encourages smooth, spatially coherent gap fills
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=pred_sq.dtype, device=pred_sq.device,
    ).view(1, 1, 3, 3)

    pred_lap = F.conv2d(
        pred_sq.unsqueeze(1), laplacian_kernel, padding=1
    ).squeeze(1)
    target_lap = F.conv2d(
        target_f.unsqueeze(1), laplacian_kernel, padding=1
    ).squeeze(1)

    grad_diff = (pred_lap - target_lap).pow(2)
    grad_loss = (grad_diff * sup_mask).sum() / n_valid

    # [v3] L1 bias penalty — directly penalizes systematic over/under-prediction
    # on gap pixels (attacks the GAP_BIAS issue at the source)
    bias = ((pred_sq - target_f) * sup_mask).sum() / n_valid
    bias_loss = bias.abs()

    # [v3] Laplacian weight raised from 0.02 → 0.05 (stable with grad clip 1.0)
    # Was 0.1 in early v2, caused NaN at epoch 73; 0.05 is the safe middle ground
    return nll_loss + 0.05 * grad_loss + 0.1 * bias_loss


# ======================================================================
# Forecast loss — masked Huber (smooth-L1)
# ======================================================================

def forecast_loss(
    pred: Tensor,
    target: Tensor,
    target_mask: Tensor,
    land_mask: Tensor,
    delta: float = 0.5,
    ssim_weight: float = 0.2,
) -> Tensor:
    """
    Masked Huber (smooth-L1) + SSIM loss for Chl-a forecasting.

    [v3] Added differentiable SSIM component to improve structural
    similarity of forecast maps. Weight 0.2 by default.
    """
    ocean = (1.0 - land_mask).unsqueeze(1)
    valid = target_mask * ocean

    huber = F.huber_loss(pred, target, reduction="none", delta=delta)
    n_valid = valid.sum().clamp(min=1.0)
    l_huber = (huber * valid).sum() / n_valid

    # [v3] SSIM component — per-step, averaged
    if ssim_weight > 0.0:
        window = _gaussian_window(7, 1.5).to(pred.device, pred.dtype)
        B, S, H, W = pred.shape
        ssim_total = 0.0
        for s in range(S):
            p = (pred[:, s:s+1] * valid[:, s:s+1]).float()
            t = (target[:, s:s+1] * valid[:, s:s+1]).float()
            smap = _ssim_map(p, t, window)
            ssim_total = ssim_total + (1.0 - smap).mean()
        l_ssim = ssim_total / S
        return l_huber + ssim_weight * l_ssim

    return l_huber


# ======================================================================
# ERI loss — ordinal cross-entropy
# ======================================================================

def eri_loss(
    logits: Tensor,
    target: Tensor,
    land_mask: Tensor,
    bloom_mask: Tensor | None = None,
    focal_gamma: float = 2.0,
) -> Tensor:
    """
    Ordinal cross-entropy for ERI classification (5 levels: 0-4).
    Focal modulation + soft ordinal penalty + rebalanced class weights.
    """
    B, n_levels, H, W = logits.shape
    ocean = 1.0 - land_mask

    log_probs = F.log_softmax(logits, dim=1)
    probs     = log_probs.exp()

    target_long = target.long().clamp(0, n_levels - 1)
    nll = F.nll_loss(log_probs, target_long, reduction="none")

    # [v3] class 1 weight 10.0 → 15.0 (was 5.0 in v1) for 0.03% imbalance
    class_weights = torch.tensor(
        [0.15, 15.0, 4.0, 4.0, 5.0], device=logits.device
    )
    sample_weight = class_weights[target_long]

    level_idx  = torch.arange(n_levels, device=logits.device, dtype=torch.float)
    level_idx  = level_idx.view(1, n_levels, 1, 1)
    soft_level = (probs * level_idx).sum(dim=1)
    ord_penalty = 1.0 + (soft_level - target.float()).abs()

    p_correct = probs.gather(
        dim=1, index=target_long.unsqueeze(1)
    ).squeeze(1)
    focal_weight = (1.0 - p_correct).clamp(min=0.0).pow(focal_gamma)

    pixel_loss = nll * ord_penalty * sample_weight * focal_weight

    weight = ocean.clone()
    if bloom_mask is not None:
        bloom_any = (bloom_mask.sum(dim=1) > 0).float()
        weight = weight * (1.0 + 5.0 * bloom_any)

    n_valid = weight.sum().clamp(min=1.0)
    return (pixel_loss * weight).sum() / n_valid


# ======================================================================
# ERI target builder
# ======================================================================

def build_eri_target(bloom_mask: Tensor) -> Tensor:
    """
    Derive integer ERI target labels (0-4) from bloom_mask.
    """
    bloom_count = bloom_mask.sum(dim=1)

    eri = torch.zeros_like(bloom_count, dtype=torch.long)
    eri[bloom_count >= 1]  = 1
    eri[bloom_count >= 3]  = 2
    eri[bloom_count >= 5]  = 3
    eri[bloom_count >= 8]  = 4

    return eri


# ======================================================================
# [FEAT 1] Bloom forecast loss — binary CE for per-step bloom prediction
# ======================================================================

def bloom_forecast_loss(
    logits: Tensor,
    target_chl: Tensor,
    target_mask: Tensor,
    land_mask: Tensor,
    bloom_threshold: float = 0.0,
    pos_weight_value: float = 20.0,
) -> Tensor:
    """
    Binary cross-entropy loss for elevated Chl-a prediction at each forecast step.

    Targets are derived from target_chl: a pixel is "elevated" if its
    z-scored Chl-a exceeds bloom_threshold at that forecast step.
    With the default threshold=2.5, this detects anomalously high Chl-a
    (≈0.94 mg/m³ raw), not true algal blooms (≥10 mg/m³).

    Args:
        logits:         (B, H_fcast, H, W)  raw logits from BloomForecastHead
        target_chl:     (B, H_fcast, H, W)  future Chl-a (normalized)
        target_mask:    (B, H_fcast, H, W)  1 = valid supervisable pixel
        land_mask:      (B, H, W)           1 = land
        bloom_threshold: threshold on z-scored Chl-a for "elevated" detection.
                        Train.py passes 2.5, which in z-score space corresponds
                        to raw Chl-a ≈ exp(2.5*0.2077 + 0.1450) - 1 ≈ 0.94 mg/m³.
                        This predicts "elevated Chl-a anomalies" (top ~0.6% of
                        pixels), NOT true algal blooms (≥10 mg/m³). The true
                        bloom threshold in z-score space would be
                        (log1p(10.0) - 0.1450) / 0.2077 ≈ 10.85, which produces
                        near-zero positives and is impractical for training.
                        NOTE: This is intentionally different from bloom_mask
                        (masker.py, raw 10 mg/m³) used for ERI supervision.
        pos_weight_value: positive class weight for BCE. Elevated events are rare
                         (~0.5% of pixels), so we upweight positives heavily.

    Returns:
        Scalar loss.
    """
    ocean = (1.0 - land_mask).unsqueeze(1)           # (B, 1, H, W)
    valid = target_mask * ocean                       # (B, H_fcast, H, W)

    # Build binary bloom target from forecast Chl-a
    bloom_target = (target_chl > bloom_threshold).float()  # (B, H_fcast, H, W)

    # Positive class weight — blooms are very rare
    pos_weight = torch.tensor(
        [pos_weight_value], device=logits.device
    )

    # Element-wise BCE with logits
    bce = F.binary_cross_entropy_with_logits(
        logits, bloom_target, reduction="none",
        pos_weight=pos_weight,
    )

    n_valid = valid.sum().clamp(min=1.0)
    return (bce * valid).sum() / n_valid


def build_bloom_forecast_target(
    target_chl: Tensor,
    bloom_threshold: float = 0.0,
) -> Tensor:
    """
    Convenience: binary bloom labels at each forecast step.

    Args:
        target_chl:      (B, H_fcast, H, W) normalized future Chl-a
        bloom_threshold:  threshold on normalized scale

    Returns:
        (B, H_fcast, H, W) binary, 1 = bloom at that step
    """
    return (target_chl > bloom_threshold).float()


# ======================================================================
# Combined loss
# ======================================================================

@dataclass
class LossWeights:
    recon:       float = 1.0
    forecast:    float = 0.5
    eri:         float = 0.3
    bloom_fcast: float = 0.3    # [FEAT 1] bloom lead-time prediction
    aux:         float = 0.01   # [PERF 3] was 0.001 — fix expert collapse
    holdout:     float = 0.8    # [v3] was 0.5 — stronger gap supervision


class MARASSLoss(nn.Module):
    """
    Combined MM-MARAS training loss (v3).

    Changes from v2:
        - holdout loss: Laplacian weight 0.02→0.05, added L1 bias term
        - forecast loss: added SSIM component (weight 0.2)
        - ERI: class 1 weight 10→15
        - holdout weight 0.5→0.8
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        curriculum_frac: float = 0.20,
        forecast_delta: float = 0.5,
        eri_focal_gamma: float = 2.0,
        bloom_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.curriculum_frac  = curriculum_frac
        self.forecast_delta   = forecast_delta
        self.eri_focal_gamma  = eri_focal_gamma
        self.bloom_threshold  = bloom_threshold

    def _curriculum_scale(self, step: int | None, total_steps: int | None) -> float:
        if step is None or total_steps is None:
            return 1.0
        warmup = int(total_steps * self.curriculum_frac)
        if warmup == 0:
            return 1.0
        return min(1.0, step / warmup)

    def forward(
        self,
        outputs: dict,
        batch: dict,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Args:
            outputs:     Model output dict (from MARASSModel.forward).
            batch:       Dataset batch dict (from MARASSDataset).
            step:        Current global training step (for curriculum).
            total_steps: Total training steps (for curriculum).

        Returns:
            loss:      Scalar total loss.
            breakdown: Dict of individual loss values for logging.
        """
        land_mask   = batch["land_mask"]
        obs_mask    = batch["obs_mask"]
        chl_obs     = batch["chl_obs"]
        target_chl  = batch["target_chl"]
        target_mask = batch["target_mask"]
        bloom_mask  = batch["bloom_mask"]

        # --- Reconstruction ---
        l_recon = recon_loss(
            pred         = outputs["recon"],
            log_var      = outputs["uncertainty"],
            target       = chl_obs,
            obs_mask     = obs_mask,
            land_mask    = land_mask,
            holdout_mask = outputs.get("holdout_mask"),
        )

        # --- Forecast (Huber) ---
        l_forecast = forecast_loss(
            pred        = outputs["forecast"],
            target      = target_chl,
            target_mask = target_mask,
            land_mask   = land_mask,
            delta       = self.forecast_delta,
        )

        # --- ERI (focal + soft ordinal) ---
        eri_target = build_eri_target(bloom_mask)
        l_eri = eri_loss(
            logits      = outputs["eri"],
            target      = eri_target,
            land_mask   = land_mask,
            bloom_mask  = bloom_mask,
            focal_gamma = self.eri_focal_gamma,
        )

        # --- [FEAT 1] Bloom forecast (binary CE) ---
        if "bloom_forecast" in outputs:
            l_bloom_fcast = bloom_forecast_loss(
                logits          = outputs["bloom_forecast"],
                target_chl      = target_chl,
                target_mask     = target_mask,
                land_mask       = land_mask,
                bloom_threshold = self.bloom_threshold,
            )
        else:
            l_bloom_fcast = torch.tensor(0.0, device=l_recon.device)

        # --- Aux (MoE load-balancing) ---
        if "routing_weights" in outputs:
            l_aux = compute_aux_loss(outputs["routing_weights"])
        else:
            l_aux = torch.tensor(0.0, device=l_recon.device)

        # --- Holdout reconstruction (NLL + gradient matching) ---
        if "holdout_mask" in outputs and outputs["holdout_mask"] is not None:
            l_holdout = holdout_recon_loss(
                pred         = outputs["recon"],
                log_var      = outputs["uncertainty"],
                target       = chl_obs[:, -1],
                holdout_mask = outputs["holdout_mask"],
                land_mask    = land_mask,
            )
        else:
            l_holdout = torch.tensor(0.0, device=l_recon.device)

        # --- Curriculum scaling for secondary tasks ---
        scale = self._curriculum_scale(step, total_steps)

        # [v3.1] Holdout loss also ramps with curriculum to prevent gradient
        # conflict between heavy gap reconstruction and secondary tasks.
        # Holdout uses sqrt(scale) so it reaches full weight faster than
        # secondary tasks (gap-fill is prerequisite for good forecasts).
        holdout_scale = scale ** 0.5  # reaches 0.84 when scale=0.70

        total = (
            self.w.recon       * l_recon
          + self.w.forecast    * scale * l_forecast
          + self.w.eri         * scale * l_eri
          + self.w.bloom_fcast * scale * l_bloom_fcast   # [FEAT 1]
          + self.w.aux         * l_aux                    # [PERF 3] 0.01
          + self.w.holdout     * holdout_scale * l_holdout
        )

        breakdown = {
            "total":       total.item(),
            "recon":       l_recon.item(),
            "forecast":    l_forecast.item(),
            "eri":         l_eri.item(),
            "bloom_fcast": l_bloom_fcast.item(),
            "aux":         l_aux.item(),
            "holdout":     l_holdout.item(),
            "curriculum_scale": scale,
        }

        return total, breakdown


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python loss.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from model import MARASSModel, ModelConfig

    cfg   = ModelConfig()
    model = MARASSModel(cfg)
    model.train()

    B = 4
    T, H, W, H_fcast = cfg.T, cfg.H, cfg.W, cfg.H_fcast

    obs = (torch.rand(B, T, H, W) > 0.30).float()
    batch = {
        "chl_obs":     torch.randn(B, T, H, W) * obs,
        "obs_mask":    obs,
        "mcar_mask":   torch.zeros(B, T, H, W),
        "mnar_mask":   torch.zeros(B, T, H, W),
        "bloom_mask":  (torch.rand(B, T, H, W) > 0.95).float(),
        "physics":     torch.randn(B, T, 6, H, W),
        "wind":        torch.randn(B, T, 4, H, W),
        "static":      torch.randn(B, 2, H, W),
        "discharge":   torch.randn(B, T, 2, H, W),
        "bgc_aux":     torch.randn(B, T, 5, H, W),
        "land_mask":   (torch.rand(B, H, W) > 0.97).float(),
        "target_chl":  torch.randn(B, H_fcast, H, W),
        "target_mask": (torch.rand(B, H_fcast, H, W) > 0.30).float(),
    }

    outputs = model(batch)
    criterion = MARASSLoss(bloom_threshold=0.0)

    # Step 0 — curriculum should be 0 for forecast/ERI/bloom
    loss, breakdown = criterion(outputs, batch, step=0, total_steps=1000)

    print("\n--- Loss breakdown (step=0, curriculum=0%) ---")
    for k, v in breakdown.items():
        print(f"  {k:<22} {v:.4f}")

    assert torch.isfinite(loss), "Loss is not finite"
    print(f"\nTotal loss: {loss.item():.4f}  (finite: OK)")

    # Verify bloom_fcast loss is present
    assert "bloom_fcast" in breakdown, "bloom_fcast missing from breakdown"
    print(f"Bloom forecast loss: {breakdown['bloom_fcast']:.4f}")

    # Backward pass
    loss.backward()
    n_grads = sum(1 for _, p in model.named_parameters() if p.grad is not None)
    print(f"\nBackward pass: {n_grads} params with gradients")

    # Bloom forecast target builder
    bloom_tgt = build_bloom_forecast_target(batch["target_chl"], bloom_threshold=0.0)
    print(f"Bloom forecast target: {tuple(bloom_tgt.shape)}, "
          f"{bloom_tgt.mean().item()*100:.1f}% positive")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    run_smoke_test()