"""Pydantic v2 schemas for subscriptions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# GeoJSON helpers
# ──────────────────────────────────────────────────────────────────────────────

class GeoJSONPolygon(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]   # [[[lon, lat], ...]]

    @model_validator(mode="after")
    def _check_ring_closed(self) -> "GeoJSONPolygon":
        for ring in self.coordinates:
            if len(ring) < 4:
                raise ValueError("Polygon ring must have at least 4 positions (first == last).")
            if ring[0] != ring[-1]:
                raise ValueError("Polygon ring must be closed (first position == last).")
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Subscription CRUD
# ──────────────────────────────────────────────────────────────────────────────

class SubscriptionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    contact_email: EmailStr
    contact_phone: Optional[str] = None
    geometry: GeoJSONPolygon
    severity_threshold: float = Field(0.5, ge=0.0, le=1.0)
    channels: list[Literal["email", "sms", "inapp"]] = Field(default_factory=lambda: ["email"])


class SubscriptionOut(BaseModel):
    id: uuid.UUID
    name: str
    contact_email: str
    contact_phone: Optional[str]
    severity_threshold: float
    channels: list[str]
    confirmed_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class SubscriptionPending(BaseModel):
    id: uuid.UUID
    status: Literal["pending_confirmation"] = "pending_confirmation"
