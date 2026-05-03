"""
app/worker/ingest.py
--------------------
Download a recent window of CMEMS / ERA5 / GloFAS data into
<data_dir>/raw/<target_date>/, then rename outputs to the canonical names
that daily_run.py consumes:

    cmems_chl.nc, cmems_phys.nc, era5_wind.nc, era5_msl.nc,
    era5_precip.nc, glofas.nc

Wraps loader functions from data-preprocessing-pipeline/loader.py.

Auth
----
CMEMS  : username/password passed as kwargs (read from CMEMS_USERNAME /
         CMEMS_PASSWORD env vars).
CDS    : cdsapi auto-reads CDSAPI_URL and CDSAPI_KEY from the environment.
         GloFAS uses a different endpoint (hardcoded inside loader.py) but
         the same key.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

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
    status:           str           = "ok"   # ok | failed
    attempts:         int           = 1
    bytes_written:    Optional[int] = None
    remote_latest_ts: Optional[str] = None
    message:          Optional[str] = None
    local_path:       Optional[str] = None


@dataclass
class IngestResult:
    target_date: date
    raw_dir:     Path
    sources:     list[SourceResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(s.status == "ok" for s in self.sources)

    def summary_dict(self) -> dict:
        return {s.source: {"status": s.status, "bytes": s.bytes_written,
                           "latest_ts": s.remote_latest_ts, "msg": s.message}
                for s in self.sources}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _date_window(target_date: date, time_window: int) -> tuple[str, str]:
    """ISO (start, end) covering target_date - (T+2) ... target_date inclusive."""
    start = target_date - timedelta(days=time_window + 2)
    return str(start), str(target_date)


def _ensure_canonical(
    raw_dir: Path,
    canonical_name: str,
    download_fn: Callable[[], Path],
) -> Path:
    """
    If <raw_dir>/<canonical_name> already exists, skip the download.
    Otherwise call download_fn() (which writes a file under a loader-native
    name into raw_dir) and rename the result to canonical_name.
    """
    canonical_path = raw_dir / canonical_name
    if canonical_path.exists():
        log.info(f"Canonical file present, skipping: {canonical_path.name}")
        return canonical_path
    actual_path = Path(download_fn())
    if actual_path.resolve() != canonical_path.resolve():
        actual_path.replace(canonical_path)
    return canonical_path


def _cmems_creds() -> tuple[Optional[str], Optional[str]]:
    return os.environ.get("CMEMS_USERNAME"), os.environ.get("CMEMS_PASSWORD")


# ──────────────────────────────────────────────────────────────────────────────
# Per-source download helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download_cmems_chl(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_copernicus  # type: ignore[import]
    import config as cfg                    # type: ignore[import]

    domain     = cfg.DOMAINS[cfg.ACTIVE_DOMAIN]
    user, pw   = _cmems_creds()
    start, end = _date_window(target_date, cfg.TIME_WINDOW)
    result     = SourceResult(source="cmems_chl")

    try:
        path = _ensure_canonical(
            raw_dir, "cmems_chl.nc",
            lambda: download_copernicus(
                dataset_id = cfg.COPERNICUS_BGC_DATASET,
                variables  = cfg.BGC_VARIABLES,
                date_start = start,
                date_end   = end,
                lon_min    = domain["lon_min"],
                lon_max    = domain["lon_max"],
                lat_min    = domain["lat_min"],
                lat_max    = domain["lat_max"],
                depth_min  = cfg.BGC_DEPTH_MIN,
                depth_max  = cfg.BGC_DEPTH_MAX,
                output_dir = str(raw_dir),
                username   = user,
                password   = pw,
            ),
        )
        result.local_path    = str(path)
        result.bytes_written = path.stat().st_size
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


def _download_cmems_phys(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_copernicus  # type: ignore[import]
    import config as cfg                    # type: ignore[import]

    domain     = cfg.DOMAINS[cfg.ACTIVE_DOMAIN]
    user, pw   = _cmems_creds()
    start, end = _date_window(target_date, cfg.TIME_WINDOW)
    result     = SourceResult(source="cmems_phys")

    try:
        path = _ensure_canonical(
            raw_dir, "cmems_phys.nc",
            lambda: download_copernicus(
                dataset_id = cfg.COPERNICUS_PHY_DATASET,
                variables  = cfg.PHY_VARIABLES,
                date_start = start,
                date_end   = end,
                lon_min    = domain["lon_min"],
                lon_max    = domain["lon_max"],
                lat_min    = domain["lat_min"],
                lat_max    = domain["lat_max"],
                depth_min  = cfg.PHY_DEPTH_MIN,
                depth_max  = cfg.PHY_DEPTH_MAX,
                output_dir = str(raw_dir),
                username   = user,
                password   = pw,
            ),
        )
        result.local_path    = str(path)
        result.bytes_written = path.stat().st_size
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


def _download_era5(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import (  # type: ignore[import]
        download_era5_wind, download_era5_msl, download_era5_precip,
    )
    import config as cfg  # type: ignore[import]

    domain     = cfg.DOMAINS[cfg.ACTIVE_DOMAIN]
    start, end = _date_window(target_date, cfg.TIME_WINDOW)
    bounds = dict(
        date_start = start,
        date_end   = end,
        lon_min    = domain["lon_min"],
        lon_max    = domain["lon_max"],
        lat_min    = domain["lat_min"],
        lat_max    = domain["lat_max"],
        output_dir = str(raw_dir),
    )

    result = SourceResult(source="era5")
    total  = 0
    try:
        for fn, canonical in [
            (download_era5_wind,   "era5_wind.nc"),
            (download_era5_msl,    "era5_msl.nc"),
            (download_era5_precip, "era5_precip.nc"),
        ]:
            path = _ensure_canonical(raw_dir, canonical, lambda fn=fn: fn(**bounds))
            total += path.stat().st_size
        result.bytes_written = total
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


def _download_glofas(raw_dir: Path, target_date: date) -> SourceResult:
    _add_pipeline_path()
    from loader import download_glofas  # type: ignore[import]
    import config as cfg                # type: ignore[import]

    domain     = cfg.DOMAINS[cfg.ACTIVE_DOMAIN]
    start, end = _date_window(target_date, cfg.TIME_WINDOW)
    result     = SourceResult(source="glofas")

    try:
        path = _ensure_canonical(
            raw_dir, "glofas.nc",
            lambda: download_glofas(
                date_start = start,
                date_end   = end,
                lon_min    = domain["lon_min"],
                lon_max    = domain["lon_max"],
                lat_min    = domain["lat_min"],
                lat_max    = domain["lat_max"],
                output_dir = str(raw_dir),
            ),
        )
        result.local_path    = str(path)
        result.bytes_written = path.stat().st_size
    except Exception as exc:
        result.status  = "failed"
        result.message = str(exc)
        raise
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Retry wrappers
# ──────────────────────────────────────────────────────────────────────────────

def _with_retry(fn, raw_dir: Path, target_date: date) -> SourceResult:
    """Run fn up to 3 times with exponential backoff, tracking attempt count."""
    attempt = 0

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=2, min=10, max=120),
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
    Returns IngestResult regardless of failures — caller decides how to proceed.
    """
    raw_dir = data_dir / "raw" / str(target_date)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Ingesting data for {target_date} → {raw_dir}")

    result = IngestResult(target_date=target_date, raw_dir=raw_dir)
    for fn in (_download_cmems_chl, _download_cmems_phys, _download_era5, _download_glofas):
        src = _with_retry(fn, raw_dir, target_date)
        result.sources.append(src)
        if src.status == "ok":
            log.info(f"  {src.source}: ok  bytes={src.bytes_written}")
        else:
            log.error(f"  {src.source}: FAILED after {src.attempts} attempts — {src.message}")

    return result
