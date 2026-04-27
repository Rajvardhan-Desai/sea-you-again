"""
/api/forecast/* — public forecast endpoints.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from shapely.geometry import shape
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.schemas.forecast import AOISummaryOut, AOISummaryRequest, ForecastOut, LayerURLs, TimeSeriesPoint
from app.api.settings import Settings, get_settings
from app.db.models import Run

router = APIRouter(prefix="/forecast", tags=["forecast"])


def _build_forecast_out(run: Run, settings: Settings) -> ForecastOut:
    """Build ForecastOut from a Run ORM object."""
    if not run.artifacts_path:
        raise HTTPException(status_code=404, detail="Artifacts not yet available for this run.")
    art  = Path(run.artifacts_path)
    meta_path = art / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json missing — run may have failed.")

    with open(meta_path) as f:
        meta = json.load(f)

    base_url = f"{settings.public_base_url}/static/runs/{run.run_date}"
    layers = meta.get("layers", {})

    def _urls(layer: str) -> list[str]:
        return [f"{base_url}/overlays/{name}" for name in layers.get(layer, [])]

    return ForecastOut(
        run_id           = run.id,
        run_date         = run.run_date,
        horizons         = meta.get("horizons", []),
        layers           = LayerURLs(
            recon    = _urls("recon"),
            forecast = _urls("forecast"),
            bloom    = _urls("bloom"),
            eri      = _urls("eri"),
            impact   = _urls("impact"),
        ),
        overlay_base_url = f"{base_url}/overlays",
        bbox             = meta.get("bbox", []),
        crs              = meta.get("crs", "EPSG:4326"),
        inference_metrics = run.inference_metrics,
    )


@router.get("/latest", response_model=ForecastOut)
def get_latest_forecast(
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ForecastOut:
    run = (
        db.query(Run)
        .filter(Run.status == "succeeded")
        .order_by(desc(Run.run_date))
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="No successful runs available yet.")
    return _build_forecast_out(run, settings)


@router.get("/runs/{run_id}", response_model=ForecastOut)
def get_forecast_by_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ForecastOut:
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return _build_forecast_out(run, settings)


@router.get("/timeseries", response_model=list[TimeSeriesPoint])
def get_timeseries(
    lat:   float = Query(..., ge=-90,  le=90),
    lon:   float = Query(..., ge=-180, le=180),
    layer: str   = Query("forecast", pattern="^(forecast|bloom|eri|impact|recon)$"),
    days:  int   = Query(30, ge=1, le=365),
    db:    Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> list[TimeSeriesPoint]:
    """
    Return a per-day time series of the mean pixel value at (lat, lon)
    for the given layer across the last `days` successful runs.
    This reads the first horizon (d1) NPZ artifact to extract the value.
    """
    since = date.today() - timedelta(days=days)
    runs  = (
        db.query(Run)
        .filter(Run.status == "succeeded", Run.run_date >= since)
        .order_by(Run.run_date)
        .all()
    )

    data_dir = Path(settings.data_dir)
    points: list[TimeSeriesPoint] = []

    for run in runs:
        if not run.artifacts_path:
            continue
        art = Path(run.artifacts_path)
        # Load metadata to get bbox
        meta_path = art / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        bbox = meta.get("bbox")  # [minlon, minlat, maxlon, maxlat]
        if not bbox:
            continue

        npz_path = art / f"{layer}.npz"
        if not npz_path.exists():
            continue

        arr_data = np.load(npz_path)
        # NPZ stores (5, H, W) for forecast/bloom/eri, (H, W) for recon/impact
        key = list(arr_data.files)[0]
        arr = arr_data[key]
        if arr.ndim == 3:
            arr = arr[0]  # first horizon

        H, W = arr.shape
        # Map lat/lon to pixel index (assume regular grid)
        minlon, minlat, maxlon, maxlat = bbox
        if not (minlat <= lat <= maxlat and minlon <= lon <= maxlon):
            continue

        col = int((lon - minlon) / (maxlon - minlon) * (W - 1))
        row = int((maxlat - lat) / (maxlat - minlat) * (H - 1))
        val = float(arr[row, col])
        if np.isfinite(val):
            points.append(TimeSeriesPoint(date=str(run.run_date), value=val))

    return points


@router.post("/aoi-summary", response_model=AOISummaryOut)
def get_aoi_summary(
    body: AOISummaryRequest,
    db:   Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AOISummaryOut:
    """Compute per-horizon stats for a GeoJSON polygon over the latest run."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.mask import mask as rio_mask

    run = (
        db.query(Run)
        .filter(Run.status == "succeeded")
        .order_by(desc(Run.run_date))
        .first()
    )
    if not run or not run.artifacts_path:
        raise HTTPException(status_code=404, detail="No successful runs available yet.")

    art = Path(run.artifacts_path)
    meta_path = art / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json missing.")

    with open(meta_path) as f:
        meta = json.load(f)
    bbox = meta.get("bbox")  # [minlon, minlat, maxlon, maxlat]
    if not bbox:
        raise HTTPException(status_code=500, detail="No bbox in metadata.")

    minlon, minlat, maxlon, maxlat = bbox
    poly = shape(body.geometry)

    def _load_and_mask(npz_path: Path, n_bands: int) -> np.ndarray:
        """Load npz, build in-memory rasterio dataset, mask with polygon."""
        arr_data = np.load(npz_path)
        key = list(arr_data.files)[0]
        arr = arr_data[key]   # (5, H, W) or (H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis]   # (1, H, W)

        H, W = arr.shape[-2], arr.shape[-1]
        transform = rasterio.transform.from_bounds(minlon, minlat, maxlon, maxlat, W, H)

        max_bloom_prob = []
        max_eri        = []
        mean_chl       = []
        pixel_count    = 0

        for h in range(min(n_bands, arr.shape[0])):
            band = arr[h].astype(np.float32)
            profile = {
                "driver": "GTiff", "dtype": "float32",
                "width": W, "height": H, "count": 1,
                "crs": "EPSG:4326", "transform": transform,
            }
            with MemoryFile() as memfile:
                with memfile.open(**profile) as ds:
                    ds.write(band, 1)
                with memfile.open() as ds:
                    try:
                        masked, _ = rio_mask(ds, [poly], crop=True, nodata=np.nan)
                        vals = masked[0][np.isfinite(masked[0])]
                    except Exception:
                        vals = np.array([])
            if len(vals) > 0:
                pixel_count = max(pixel_count, len(vals))
                max_bloom_prob.append(float(np.max(vals)))
                mean_chl.append(float(np.mean(vals)))
            else:
                max_bloom_prob.append(0.0)
                mean_chl.append(0.0)

        return max_bloom_prob, mean_chl, pixel_count

    # bloom
    bloom_path = art / "bloom.npz"
    if not bloom_path.exists():
        raise HTTPException(status_code=404, detail="bloom.npz missing.")
    max_bloom, _, pixel_count = _load_and_mask(bloom_path, 5)

    # eri
    eri_path = art / "eri.npz"
    eri_max_per_h: list[int] = [0] * 5
    if eri_path.exists():
        arr_data = np.load(eri_path)
        eri_arr  = arr_data[list(arr_data.files)[0]]
        H, W     = eri_arr.shape[-2], eri_arr.shape[-1]
        transform = rasterio.transform.from_bounds(minlon, minlat, maxlon, maxlat, W, H)
        for h in range(min(5, eri_arr.shape[0])):
            band = eri_arr[h].astype(np.float32)
            profile = {"driver": "GTiff", "dtype": "float32", "width": W, "height": H,
                       "count": 1, "crs": "EPSG:4326", "transform": transform}
            with MemoryFile() as mf:
                with mf.open(**profile) as ds:
                    ds.write(band, 1)
                with mf.open() as ds:
                    try:
                        masked, _ = rio_mask(ds, [poly], crop=True, nodata=np.nan)
                        vals = masked[0][np.isfinite(masked[0])]
                        eri_max_per_h[h] = int(np.max(vals)) if len(vals) > 0 else 0
                    except Exception:
                        pass

    # chl forecast
    fcast_path = art / "forecast.npz"
    mean_chl_per_h = [0.0] * 5
    if fcast_path.exists():
        _, mean_chl_per_h, _ = _load_and_mask(fcast_path, 5)

    return AOISummaryOut(
        max_bloom_prob_per_horizon = max_bloom,
        max_eri_per_horizon        = eri_max_per_h,
        mean_chl_per_horizon       = mean_chl_per_h,
        pixel_count                = pixel_count,
    )
