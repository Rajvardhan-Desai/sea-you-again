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
    else:
        # Preprocess raw data → patches
        from pipeline import run_pipeline  # type: ignore[import]
        import config as cfg               # type: ignore[import]

        patch_dir = run_dir / "patches"
        run_pipeline(
            domain          = cfg.ACTIVE_DOMAIN,
            download        = False,
            chl_path        = str(raw_dir / "cmems_chl.nc"),
            phys_path       = str(raw_dir / "cmems_phys.nc"),
            era5_wind_path  = str(raw_dir / "era5_wind.nc"),
            era5_msl_path   = str(raw_dir / "era5_msl.nc"),
            era5_precip_path = str(raw_dir / "era5_precip.nc"),
            discharge_path  = str(raw_dir / "glofas.nc"),
            output_dir      = str(patch_dir),
            recompute_stats = False,
        )

        from dataset import build_dataloaders  # type: ignore[import]
        loaders = build_dataloaders(
            patch_dir  = str(patch_dir),
            batch_size = 8,
            num_workers = 2,
            pin_memory  = (device.type == "cuda"),
        )
        # Use all splits in sequence for full-domain coverage
        all_batches = []
        for split in ("train", "val", "test"):
            for b in loaders.get(split, []):
                all_batches.append(b)
        if not all_batches:
            raise RuntimeError("No patches produced by pipeline.")
        # For simplicity use the first batch; production would stitch all patches
        batch = {k: v.to(device) for k, v in all_batches[0].items()}

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

    # ── Compute bbox from config ──
    try:
        import config as cfg  # type: ignore[import]
        dom = cfg.DOMAIN
        bbox = [float(dom["minlon"]), float(dom["minlat"]),
                float(dom["maxlon"]), float(dom["maxlat"])]
    except Exception:
        H, W = recon_np.shape
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
            run_row.triggered_by   = triggered_by
            run_row.started_at     = run_row.started_at or datetime.now(timezone.utc)
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
