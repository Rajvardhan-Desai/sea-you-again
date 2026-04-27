"""Pydantic v2 schemas for forecast endpoints."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Optional

from pydantic import BaseModel


class LayerURLs(BaseModel):
    recon:    list[str]   # 5 URLs (d1..d5)
    forecast: list[str]
    bloom:    list[str]
    eri:      list[str]
    impact:   list[str]


class ForecastOut(BaseModel):
    run_id:          uuid.UUID
    run_date:        date
    horizons:        list[str]     # 5 ISO date strings
    layers:          LayerURLs
    overlay_base_url: str
    bbox:            list[float]   # [minlon, minlat, maxlon, maxlat]
    crs:             str = "EPSG:4326"
    inference_metrics: Optional[dict[str, Any]] = None


class TimeSeriesPoint(BaseModel):
    date:  str   # ISO
    value: float


class AOISummaryRequest(BaseModel):
    geometry: dict   # GeoJSON Polygon


class AOISummaryOut(BaseModel):
    max_bloom_prob_per_horizon: list[float]   # length 5
    max_eri_per_horizon:        list[int]
    mean_chl_per_horizon:       list[float]
    pixel_count:                int
