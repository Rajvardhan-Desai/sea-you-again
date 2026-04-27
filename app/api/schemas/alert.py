"""Pydantic v2 schemas for alerts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class AlertOut(BaseModel):
    id:               uuid.UUID
    run_id:           uuid.UUID
    severity:         Optional[float]
    max_bloom_prob:   Optional[float]
    max_eri:          Optional[int]
    horizon_of_max:   Optional[int]
    polygon_summary:  Optional[dict[str, Any]]
    sent_at:          Optional[datetime]
    channels_sent:    Optional[list[str]]

    model_config = {"from_attributes": True}
