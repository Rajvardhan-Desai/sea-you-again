"""
app/inference/serve.py
----------------------
Canonical inference forward pass for MM-MARAS, shared by:
  - scripts/eval.py  (offline evaluation)
  - app/worker/daily_run.py  (daily ingestion → inference pipeline)

This module was extracted from scripts/eval.py (lines 734-784) so that
both consumers import from a single source of truth.  The model is NOT
loaded here — see load_model.py for the singleton loader.
"""

from __future__ import annotations

import torch


def forward_with_routing(
    model: torch.nn.Module,
    batch: dict,
) -> tuple[dict, torch.Tensor]:
    """
    Run model in eval-mode forward and always collect MoE routing weights.

    Parameters
    ----------
    model : MARASSModel (or DataParallel wrapper)
    batch : dict of tensors already moved to the model's device

    Returns
    -------
    outputs : dict with keys
        recon           (B, 1, H, W)   gap-filled Chl-a
        forecast        (B, 5, H, W)   5-day ahead Chl-a
        uncertainty     (B, 1, H, W)   aleatoric log-variance
        eri             (B, 5, H, W)   ordinal ERI logits (0-4)
        bloom_forecast  (B, 5, H, W)   per-horizon bloom logits
    routing_weights : Tensor (B, n_experts)
    """
    m = model.module if hasattr(model, "module") else model
    cfg = m.cfg

    chl_obs    = batch["chl_obs"]
    obs_mask   = batch["obs_mask"]
    mcar_mask  = batch["mcar_mask"]
    mnar_mask  = batch["mnar_mask"]
    bloom_mask = batch["bloom_mask"]
    physics    = batch["physics"]
    wind       = batch["wind"]
    static_    = batch["static"]
    discharge  = batch["discharge"]
    bgc_aux    = batch["bgc_aux"]

    optical = torch.stack([chl_obs, obs_mask], dim=2)
    masks   = torch.stack([obs_mask, mcar_mask, mnar_mask, bloom_mask], dim=2)

    mask_emb = m.masknet(masks)
    opt_feat = m.opt_enc(optical)
    phy_feat = m.phy_enc(physics, wind, static_)
    bgc_feat = m.bgc_enc(bgc_aux)
    dis_feat = m.discharge_enc(discharge)

    fused = m.fusion(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)

    # v3: temporal returns (state, h_sequence)
    state, h_sequence = m.temporal(fused)

    # v3: temporal attention enriches state with sequence
    state_enriched = m.temporal_attn(state, h_sequence, obs_mask)

    decoded, routing_w = m.decoder(state_enriched, return_routing=True)

    # v3: ReconHead takes (decoded, opt_skip, obs_mask_last)
    opt_skip      = opt_feat[:, -1]
    obs_mask_last = obs_mask[:, -1]
    recon = m.recon_head(decoded, opt_skip, obs_mask_last)

    # v3.1: ERIHead takes only decoded (bloom_count removed — label leakage fix)
    eri = m.eri_head(decoded)

    return {
        "recon":          recon,
        "forecast":       m.forecast_head(decoded),
        "uncertainty":    m.uncertainty_head(decoded),
        "eri":            eri,
        "bloom_forecast": m.bloom_fcast_head(decoded),
    }, routing_w
