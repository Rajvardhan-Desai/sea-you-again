"""
app/worker/daily_run.py
-----------------------
Daily ingestion → preprocessing → inference → artifact writing → alert evaluation.

Each phase updates runs.status and is idempotent so retries are safe.

CLI
---
    python -m app.worker.daily_run [--date YYYY-MM-DD] [--triggered-by STR]
                                   [--start-phase ingest|infer|alert]
                                   [--use-fixture PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent


def _add_pipeline_paths() -> None:
    for sub in ("model", "model/encoders", "data-preprocessing-pipeline"):
        p = str(_REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus metrics (optional — don't crash if not configured)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import Counter, Gauge

    _last_success_ts  = Gauge("mmaras_last_success_ts",        "Unix timestamp of last successful run")
    _last_duration    = Gauge("mmaras_last_run_duration_seconds", "Duration of last run in seconds")
    _sub_total        = Gauge("mmaras_subscriptions_total",     "Total confirmed subscriptions")
    _alerts_total     = Counter("mmaras_alerts_dispatched_total", "Total alerts dispatched")
    _failures_total   = Counter("mmaras_run_failures_total",    "Total run failures")
    _PROMETHEUS       = True
except Exception:
    _PROMETHEUS = False


def _set_gauge(name: str, value: float) -> None:
    if not _PROMETHEUS:
        return
    g = globals().get(f"_{name}")
    if g is not None:
        g.set(value)


# ──────────────────────────────────────────────────────────────────────────────
# Phase helpers
# ──────────────────────────────────────────────────────────────────────────────

def _update_run_status(db, run, status: str, **kwargs) -> None:
    run.status = status
    for k, v in kwargs.items():
        setattr(run, k, v)
    db.commit()
    log.info(f"Run {run.id}: status → {status}")


def _persist_data_sources(db, run, ingest_result) -> None:
    from app.db.models import DataSource
    for src in ingest_result.sources:
        row = DataSource(
            run_id           = run.id,
            source           = src.source.replace("-", "_"),
            status           = src.status,
            attempts         = src.attempts,
            bytes_written    = src.bytes_written,
            message          = src.message,
        )
        db.add(row)
    db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Inference-only batch builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_inference_batch(
    mask_ds,
    normalized:   dict,
    static_arr:   np.ndarray,
    time_window:  int,
    patch_size:   int,
    phy_vars:     list,
    bgc_aux_vars: list,
    dis_vars:     list,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Build a single-sample inference batch from full-domain preprocessed arrays.

    Takes the most recent `time_window` timesteps and crops the top-left
    `patch_size × patch_size` region. NaN handling matches the training-time
    dataset loader (see data-preprocessing-pipeline/dataset.py): land_mask is
    derived from static NaNs, then static and other "silent NaN" tensors are
    filled with 0.0; chl_obs NaNs (cloud/glint) are also filled with 0.0 because
    obs_mask already encodes which pixels were missing.

    target_chl and target_mask are placeholders here — `forward_with_routing`
    does not use them, but downstream postprocessing (ecosystem_impact) reads
    `static` and `land_mask`.

    Returns
    -------
    batch       : dict[str, np.ndarray]   single-sample (no batch dim)
    crop_lats   : np.ndarray (patch_size,)  latitudes of the cropped patch
    crop_lons   : np.ndarray (patch_size,)  longitudes of the cropped patch
    """
    chl_var = normalized["chl"]["chl"]
    if "depth" in chl_var.dims:
        chl_var = chl_var.isel(depth=0, drop=True)

    T_full = chl_var.sizes["time"]
    if T_full < time_window:
        raise RuntimeError(
            f"Need {time_window} timesteps for inference but only have {T_full}. "
            f"Widen the ingest window in app/worker/ingest.py:_date_window."
        )

    H_full = chl_var.sizes["lat"]
    W_full = chl_var.sizes["lon"]
    if H_full < patch_size or W_full < patch_size:
        raise RuntimeError(
            f"Domain {H_full}×{W_full} smaller than patch size {patch_size}."
        )

    t_sl = slice(T_full - time_window, T_full)
    r_sl = slice(0, patch_size)
    c_sl = slice(0, patch_size)

    chl_np   = chl_var.values[t_sl,   r_sl, c_sl].astype(np.float32)
    obs_np   = mask_ds["obs_mask"].values  [t_sl, r_sl, c_sl].astype(np.float32)
    mcar_np  = mask_ds["mcar_mask"].values [t_sl, r_sl, c_sl].astype(np.float32)
    mnar_np  = mask_ds["mnar_mask"].values [t_sl, r_sl, c_sl].astype(np.float32)
    bloom_np = mask_ds["bloom_mask"].values[t_sl, r_sl, c_sl].astype(np.float32)

    physics_np = np.stack(
        [normalized["physics"][v].values[t_sl] for v in phy_vars],
        axis=1,
    )[:, :, r_sl, c_sl].astype(np.float32)

    wind_np = np.stack([
        normalized["wind"]["u10"].values[t_sl],
        normalized["wind"]["v10"].values[t_sl],
        normalized["wind"]["msl"].values[t_sl],
        normalized["precip"]["tp"].values[t_sl],
    ], axis=1)[:, :, r_sl, c_sl].astype(np.float32)

    discharge_np = np.stack(
        [normalized["discharge"][v].values[t_sl] for v in dis_vars],
        axis=1,
    )[:, :, r_sl, c_sl].astype(np.float32)

    bgc_aux_np = np.stack(
        [normalized["bgc_aux"][v].values[t_sl] for v in bgc_aux_vars],
        axis=1,
    )[:, :, r_sl, c_sl].astype(np.float32)

    static_cropped = static_arr[:, r_sl, c_sl].astype(np.float32)
    land_mask_2d   = np.isnan(static_cropped).any(axis=0).astype(np.float32)

    batch = {
        "chl_obs":     np.nan_to_num(chl_np,        nan=0.0),
        "obs_mask":    obs_np,
        "mcar_mask":   mcar_np,
        "mnar_mask":   mnar_np,
        "physics":     np.nan_to_num(physics_np,    nan=0.0),
        "wind":        np.nan_to_num(wind_np,       nan=0.0),
        "discharge":   np.nan_to_num(discharge_np,  nan=0.0),
        "bgc_aux":     np.nan_to_num(bgc_aux_np,    nan=0.0),
        "static":      np.nan_to_num(static_cropped, nan=0.0),
        "bloom_mask":  bloom_np,
        "target_chl":  np.zeros((5, patch_size, patch_size), dtype=np.float32),
        "land_mask":   land_mask_2d,
        "target_mask": np.ones((5, patch_size, patch_size),  dtype=np.float32),
    }

    crop_lats = chl_var.lat.values[r_sl]
    crop_lons = chl_var.lon.values[c_sl]
    return batch, crop_lats, crop_lons


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2-7: full inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_inference(
    run,
    raw_dir:     Path,
    run_dir:     Path,
    target_date: date,
    db,
    settings,
    fixture_path: Optional[str] = None,
) -> dict:
    """
    Preprocess → batch → forward → stitch → persist artifacts.
    Returns inference_metrics dict.
    """
    import torch

    _add_pipeline_paths()
    from app.inference.load_model import get_model, get_checkpoint_hash
    from app.inference.serve import forward_with_routing
    from app.inference.postprocess import bloom_probs, eri_classes, ecosystem_impact
    from app.inference.render import render_run_overlays

    import time
    t0 = time.perf_counter()

    # Load model (singleton — already warm on second call)
    model = get_model(settings.checkpoint_path)
    device = next(model.parameters()).device

    # ── Build batch ──
    if fixture_path:
        log.info(f"Using fixture batch: {fixture_path}")
        batch_np = np.load(fixture_path, allow_pickle=True)
        batch = {k: torch.from_numpy(np.array(batch_np[k])).unsqueeze(0).to(device)
                 for k in batch_np.files}
        bbox = None  # fixture: caller-provided; falls through to domain default below
        crop_lats = crop_lons = None
    else:
        # Inference-only preprocessing: skip patch extraction (which requires
        # T_INPUT + H = 15 timesteps per train/val/test split). For inference we
        # only need the most recent T_INPUT=10 timesteps fed forward through the
        # model, which then forecasts H=5 days ahead.
        from pipeline import (  # type: ignore[import]
            step_load_and_align, step_build_masks, step_normalize, step_build_static,
        )
        import config as cfg  # type: ignore[import]

        cfg.STATS_DIR = str(Path(settings.data_dir) / "stats")

        domain     = cfg.DOMAINS[cfg.ACTIVE_DOMAIN]
        stats_path = str(Path(cfg.STATS_DIR) / f"norm_stats_{cfg.ACTIVE_DOMAIN}.json")

        # Bathymetry: prefer the local GEBCO NetCDF if present (drop at
        # <data_dir>/raw/gebco_bob.nc). Falls back to None → step_build_static
        # uses cfg.BATHY_PATH or zeros placeholder.
        gebco_path = Path(settings.data_dir) / "raw" / "gebco_bob.nc"
        bathy_path = str(gebco_path) if gebco_path.exists() else None
        if bathy_path:
            log.info(f"Using bathymetry: {bathy_path}")

        paths = {
            "chl":         str(raw_dir / "cmems_chl.nc"),
            "physics":     str(raw_dir / "cmems_phys.nc"),
            "bathy":       bathy_path,
            "era5_wind":   str(raw_dir / "era5_wind.nc"),
            "era5_msl":    str(raw_dir / "era5_msl.nc"),
            "era5_precip": str(raw_dir / "era5_precip.nc"),
            "discharge":   str(raw_dir / "glofas.nc"),
        }

        aligned    = step_load_and_align(paths, domain)
        mask_ds    = step_build_masks(aligned)
        normalized = step_normalize(aligned, mask_ds, stats_path, recompute_stats=False)
        static_arr = step_build_static(aligned["chl"], paths.get("bathy"))

        batch_np, crop_lats, crop_lons = _build_inference_batch(
            mask_ds      = mask_ds,
            normalized   = normalized,
            static_arr   = static_arr,
            time_window  = cfg.TIME_WINDOW,
            patch_size   = 64,
            phy_vars     = cfg.PHY_VARIABLES,
            bgc_aux_vars = cfg.BGC_AUX_VARIABLES,
            dis_vars     = cfg.DISCHARGE_VARIABLES,
        )
        batch = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                 for k, v in batch_np.items()}

    # ── Forward pass ──
    _update_run_status(db, run, "inferring")
    model.eval()
    import torch as _torch
    with _torch.no_grad():
        outputs, routing_w = forward_with_routing(model, batch)

    # ── Post-process ──
    b_probs = bloom_probs(outputs["bloom_forecast"])  # (B, 5, H, W)
    eri_cls = eri_classes(outputs["eri"])             # (B, H, W)
    impact  = ecosystem_impact(
        outputs["bloom_forecast"], outputs["forecast"],
        outputs["uncertainty"], batch["static"], batch["land_mask"],
    )

    # Take first sample in batch for full-domain overlay
    recon_np    = outputs["recon"][0, 0].cpu().float().numpy()
    forecast_np = outputs["forecast"][0].cpu().float().numpy()   # (5, H, W)
    bloom_np    = b_probs[0].cpu().float().numpy()               # (5, H, W)
    eri_np      = eri_cls[0].unsqueeze(0).expand(5, -1, -1).cpu().numpy()  # (5, H, W)
    impact_np   = impact[0].cpu().float().numpy()                # (H, W)
    land_np     = batch["land_mask"][0].cpu().float().numpy()    # (H, W)

    # ── Save NPZ artifacts ──
    np.savez_compressed(str(run_dir / "recon.npz"),    recon=recon_np)
    np.savez_compressed(str(run_dir / "forecast.npz"), forecast=forecast_np)
    np.savez_compressed(str(run_dir / "bloom.npz"),    bloom=bloom_np)
    np.savez_compressed(str(run_dir / "eri.npz"),      eri=eri_np)
    np.savez_compressed(str(run_dir / "impact.npz"),   impact=impact_np)

    # ── Compute bbox ──
    # When we built the batch from a 64×64 crop, use that crop's lat/lon range so
    # the rendered overlay aligns with the exact pixels the model saw. Otherwise
    # fall back to the full-domain bbox (fixture path, where we don't know the
    # crop geometry).
    if crop_lats is not None and crop_lons is not None:
        bbox = [float(crop_lons.min()), float(crop_lats.min()),
                float(crop_lons.max()), float(crop_lats.max())]
    else:
        try:
            import config as cfg  # type: ignore[import]
            dom = cfg.DOMAINS[cfg.ACTIVE_DOMAIN]
            bbox = [float(dom["lon_min"]), float(dom["lat_min"]),
                    float(dom["lon_max"]), float(dom["lat_max"])]
        except Exception:
            bbox = [78.0, 5.0, 100.0, 23.0]   # Bay of Bengal default

    # ── Horizon dates ──
    horizons = [str(target_date + timedelta(days=i + 1)) for i in range(5)]

    # ── Render PNGs ──
    ckpt_hash = get_checkpoint_hash() or ""
    metadata  = render_run_overlays(
        run_dir,
        recon     = recon_np,
        forecast  = forecast_np,
        bloom     = bloom_np,
        eri       = eri_np,
        impact    = impact_np,
        land_mask = land_np,
        bbox      = bbox,
        horizons  = horizons,
        checkpoint_hash = ckpt_hash,
    )

    fwd_seconds = time.perf_counter() - t0
    inference_metrics = {
        "fwd_seconds":       round(fwd_seconds, 2),
        "mean_uncertainty":  float(outputs["uncertainty"].mean().item()),
        "max_bloom_prob":    float(b_probs.max().item()),
        "mean_impact":       float(impact.mean().item()),
        "checkpoint_hash":   ckpt_hash,
    }
    log.info(f"Inference complete in {fwd_seconds:.1f}s")
    return inference_metrics, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run(
    target_date:   date,
    triggered_by:  str = "scheduler",
    start_phase:   str = "ingest",    # ingest | infer | alert
    fixture_path:  Optional[str] = None,
) -> None:
    from app.api.settings import get_settings
    from app.db.session import SessionLocal
    from app.db.models import Run
    from app.worker.ingest import run_ingest
    from app.worker.alert_engine import run_alert_evaluation

    settings   = get_settings()
    data_dir   = Path(settings.data_dir)
    run_dir    = data_dir / "runs" / str(target_date)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(exist_ok=True)

    db = SessionLocal()
    try:
        # ── Pre-flight: upsert runs row ──
        run_row = db.query(Run).filter(Run.run_date == target_date).first()
        if run_row is None:
            run_row = Run(
                run_date       = target_date,
                status         = "pending",
                triggered_by   = triggered_by,
                started_at     = datetime.now(timezone.utc),
                artifacts_path = str(run_dir),
            )
            db.add(run_row)
            db.commit()
            db.refresh(run_row)
        else:
            # Re-trigger: reset per-attempt fields so a successful retry doesn't
            # show stale error_text or an inflated duration from the first attempt.
            run_row.triggered_by   = triggered_by
            run_row.started_at     = datetime.now(timezone.utc)
            run_row.finished_at    = None
            run_row.error_text     = None
            run_row.artifacts_path = str(run_dir)
            db.commit()

        raw_dir = data_dir / "raw" / str(target_date)

        # ── Phase: ingest ──
        if start_phase in ("ingest",):
            _update_run_status(db, run_row, "ingesting")
            ingest = run_ingest(target_date, data_dir)
            _persist_data_sources(db, run_row, ingest)
            run_row.ingest_summary = ingest.summary_dict()
            db.commit()

            if not ingest.all_ok:
                failed_sources = [s.source for s in ingest.sources if s.status != "ok"]
                log.warning(f"Some sources failed: {failed_sources} — continuing with partial data")

        # ── Phase: infer ──
        if start_phase in ("ingest", "infer"):
            inference_metrics, metadata = _run_inference(
                run_row, raw_dir, run_dir, target_date, db, settings, fixture_path
            )
            run_row.inference_metrics = inference_metrics
            db.commit()
        else:
            # Load metadata from disk for alert phase
            with open(run_dir / "metadata.json") as f:
                metadata = json.load(f)

        # ── Phase: alert ──
        if start_phase in ("ingest", "infer", "alert"):
            _update_run_status(db, run_row, "alerting")
            alert_summary = run_alert_evaluation(db, run_row.id, run_dir, metadata, settings)

        # ── Finalize ──
        run_row.finished_at = datetime.now(timezone.utc)
        run_row.status      = "succeeded"
        db.commit()

        duration = (run_row.finished_at - run_row.started_at).total_seconds()
        log.info(f"Run {run_row.id} SUCCEEDED in {duration:.0f}s")
        _set_gauge("last_success_ts", run_row.finished_at.timestamp())
        _set_gauge("last_duration",   duration)

    except Exception as exc:
        log.exception(f"Run FAILED: {exc}")
        if _PROMETHEUS:
            _failures_total.inc()
        try:
            run_row.status     = "failed"
            run_row.error_text = str(exc)
            run_row.finished_at = datetime.now(timezone.utc)
            db.commit()
        except Exception:
            pass
        raise
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MM-MARAS daily inference run")
    p.add_argument("--date",         default=None, help="YYYY-MM-DD (default: yesterday)")
    p.add_argument("--triggered-by", default="cli")
    p.add_argument("--start-phase",  default="ingest", choices=["ingest", "infer", "alert"])
    p.add_argument("--use-fixture",  default=None,
                   help="Path to .npz fixture batch — skips ingestion and preprocessing")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    args = _parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    run(
        target_date  = target,
        triggered_by = args.triggered_by,
        start_phase  = args.start_phase,
        fixture_path = args.use_fixture,
    )
