"""
app/worker/ingest.py
--------------------
Download yesterday's data from CMEMS / ERA5 / GloFAS into <data_dir>/raw/<date>/.
Wraps the existing loader functions from data-preprocessing-pipeline/loader.py.

Returns an IngestResult with per-source status information used to populate
the data_sources DB rows and the runs.ingest_summary JSON.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent


def _add_pipeline_path() -> None:
    p = str(_REPO / "data-preprocessing-pipeline")
    if p not in sys.path:
        sys.path.insert(0, p)


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceResult:
    source:           str
    status:           str          = "ok"    # ok | retrying | failed
    attempts:         int          = 1
    bytes_written:    Optional[int] = None
    remote_latest_ts: Optional[str] = None   # ISO string
    message:          Optional[str] = None
    local_path:       Optional[str] = None


@dataclass
class IngestResult:
    target_date:   date
    raw_dir:       Path
    sources:       list[SourceResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(s.status == "ok" for s in self.sources)

    def summary_dict(self) -> dict:
        return {s.source: {"status": s.status, "bytes": s.bytes_written,
                            "latest_ts": s.remote_latest_ts, "msg": s.message}
                for s in self.sources}


# ──────────────────────────────────────────────────────────────────────────────
# Per-source download helpers (each wrapped with tenacity)
# ──────────────────────────────────────────────────────────────────────────────

def _download_cmems_chl(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_copernicus  # type: ignore[import]
    import config as cfg                   # type: ignore[import]

    out_path = raw_dir / "cmems_chl.nc"
    result   = SourceResult(source="cmems_chl")
    try:
        download_copernicus(
            dataset_id  = cfg.CMEMS_BGC_DATASET,
            variables   = cfg.BGC_VARIABLES,
            domain      = cfg.DOMAIN,
            start_date  = str(target_date - timedelta(days=cfg.T_INPUT + 2)),
            end_date    = str(target_date),
            output_path = str(out_path),
        )
        result.local_path  = str(out_path)
        result.bytes_written = out_path.stat().st_size if out_path.exists() else None
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


def _download_cmems_phys(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_copernicus  # type: ignore[import]
    import config as cfg                   # type: ignore[import]

    out_path = raw_dir / "cmems_phys.nc"
    result   = SourceResult(source="cmems_phys")
    try:
        download_copernicus(
            dataset_id  = cfg.CMEMS_PHY_DATASET,
            variables   = cfg.PHY_VARIABLES,
            domain      = cfg.DOMAIN,
            start_date  = str(target_date - timedelta(days=cfg.T_INPUT + 2)),
            end_date    = str(target_date),
            output_path = str(out_path),
        )
        result.local_path    = str(out_path)
        result.bytes_written = out_path.stat().st_size if out_path.exists() else None
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


def _download_era5(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_era5_wind, download_era5_msl, download_era5_precip  # type: ignore[import]
    import config as cfg  # type: ignore[import]

    result = SourceResult(source="era5")
    total_bytes = 0
    try:
        for fn, fname in [
            (download_era5_wind,   "era5_wind.nc"),
            (download_era5_msl,    "era5_msl.nc"),
            (download_era5_precip, "era5_precip.nc"),
        ]:
            out_path = raw_dir / fname
            fn(
                domain     = cfg.DOMAIN,
                start_date = str(target_date - timedelta(days=cfg.T_INPUT + 2)),
                end_date   = str(target_date),
                output_path = str(out_path),
            )
            if out_path.exists():
                total_bytes += out_path.stat().st_size
        result.bytes_written = total_bytes
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


def _download_glofas(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_glofas  # type: ignore[import]
    import config as cfg               # type: ignore[import]

    out_path = raw_dir / "glofas.nc"
    result   = SourceResult(source="glofas")
    try:
        download_glofas(
            domain     = cfg.DOMAIN,
            start_date = str(target_date - timedelta(days=cfg.T_INPUT + 2)),
            end_date   = str(target_date),
            output_path = str(out_path),
        )
        result.local_path    = str(out_path)
        result.bytes_written = out_path.stat().st_size if out_path.exists() else None
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Retry wrappers
# ──────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, raw_dir: Path, target_date: date) -> SourceResult:
    """Run fn up to 3 times with exponential backoff, updating attempts count."""
    attempt = 0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=120),
           reraise=True)
    def _inner():
        nonlocal attempt
        attempt += 1
        return fn(raw_dir, target_date)

    try:
        result = _inner()
        result.attempts = attempt
        return result
    except Exception as exc:
        return SourceResult(
            source   = fn.__name__.replace("_download_", "").replace("_", "-"),
            status   = "failed",
            attempts = attempt,
            message  = str(exc),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_ingest(target_date: date, data_dir: Path) -> IngestResult:
    """
    Download all modalities for target_date into data_dir/raw/<date>/.
    Returns IngestResult regardless of failures — caller decides whether to abort.
    """
    raw_dir = data_dir / "raw" / str(target_date)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Ingesting data for {target_date} → {raw_dir}")

    result = IngestResult(target_date=target_date, raw_dir=raw_dir)
    for fn in (_download_cmems_chl, _download_cmems_phys, _download_era5, _download_glofas):
        src = _with_retry(fn, raw_dir, target_date)
        result.sources.append(src)
        log.info(f"  {src.source}: {src.status}  bytes={src.bytes_written}")

    return result
