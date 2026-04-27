"""Pydantic v2 schemas for admin endpoints."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel


class RunSummary(BaseModel):
    id:                uuid.UUID
    run_date:          date
    status:            str
    started_at:        Optional[datetime]
    finished_at:       Optional[datetime]
    triggered_by:      Optional[str]
    ingest_summary:    Optional[dict[str, Any]]
    inference_metrics: Optional[dict[str, Any]]
    error_text:        Optional[str]
    artifacts_path:    Optional[str]

    model_config = {"from_attributes": True}


class DataSourceOut(BaseModel):
    id:               uuid.UUID
    run_id:           uuid.UUID
    source:           str
    status:           str
    attempts:         int
    bytes_written:    Optional[int]
    remote_latest_ts: Optional[datetime]
    message:          Optional[str]

    model_config = {"from_attributes": True}


class TriggerRunRequest(BaseModel):
    run_date: Optional[date] = None   # defaults to yesterday


class AdminSessionRequest(BaseModel):
    token: str
