"""
app/db/models.py
----------------
SQLAlchemy 2.x ORM models for MM-MARAS dashboard.

Tables
------
  runs               — one row per daily inference run
  data_sources       — per-modality ingestion log per run
  subscriptions      — fishers / authority AOI subscriptions
  alerts             — alert decisions per (subscription, run)
  inapp_notifications — in-app notification records
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from geoalchemy2 import Geography
from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    Enum,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base

# ──────────────────────────────────────────────────────────────────────────────
# Enum types
# ──────────────────────────────────────────────────────────────────────────────

RunStatus = Enum(
    "pending", "ingesting", "inferring", "alerting",
    "succeeded", "failed", "partial",
    name="run_status",
)

SourceStatus = Enum("ok", "retrying", "failed", name="source_status")

SourceName = Enum(
    "cmems_chl", "cmems_phys", "era5", "glofas",
    name="source_name",
)


# ──────────────────────────────────────────────────────────────────────────────
# runs
# ──────────────────────────────────────────────────────────────────────────────

class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID]          = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_date: Mapped[date]         = mapped_column(Date, unique=True, nullable=False)
    status: Mapped[str]            = mapped_column(RunStatus, nullable=False, default="pending")
    started_at: Mapped[Optional[datetime]]  = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    ingest_summary: Mapped[Optional[dict]]  = mapped_column(JSON, nullable=True)
    inference_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    artifacts_path: Mapped[Optional[str]]   = mapped_column(Text, nullable=True)
    error_text: Mapped[Optional[str]]       = mapped_column(Text, nullable=True)
    triggered_by: Mapped[Optional[str]]     = mapped_column(String(64), nullable=True)

    data_sources: Mapped[list["DataSource"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]]            = relationship(back_populates="run", cascade="all, delete-orphan")


# ──────────────────────────────────────────────────────────────────────────────
# data_sources
# ──────────────────────────────────────────────────────────────────────────────

class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str]   = mapped_column(SourceName, nullable=False)
    status: Mapped[str]   = mapped_column(SourceStatus, nullable=False, default="ok")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    bytes_written: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    remote_latest_ts: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped["Run"] = relationship(back_populates="data_sources")


# ──────────────────────────────────────────────────────────────────────────────
# subscriptions
# ──────────────────────────────────────────────────────────────────────────────

class Subscription(Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint("severity_threshold >= 0 AND severity_threshold <= 1",
                        name="ck_severity_range"),
        Index("ix_subscriptions_geometry", "geometry", postgresql_using="gist"),
    )

    id: Mapped[uuid.UUID]         = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str]             = mapped_column(Text, nullable=False)
    contact_email: Mapped[str]    = mapped_column(Text, nullable=False)   # CITEXT via migration
    contact_phone: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    geometry: Mapped[object]      = mapped_column(
        Geography(geometry_type="POLYGON", srid=4326), nullable=False
    )
    severity_threshold: Mapped[float] = mapped_column(nullable=False, default=0.5)
    channels: Mapped[list[str]]   = mapped_column(ARRAY(Text), nullable=False)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    unsubscribe_token: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    created_at: Mapped[datetime]   = mapped_column(nullable=False, server_default=func.now())

    alerts: Mapped[list["Alert"]] = relationship(back_populates="subscription", cascade="all, delete-orphan")
    inapp_notifications: Mapped[list["InAppNotification"]] = relationship(
        back_populates="subscription", cascade="all, delete-orphan"
    )


# ──────────────────────────────────────────────────────────────────────────────
# alerts
# ──────────────────────────────────────────────────────────────────────────────

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("subscription_id", "run_id", name="uq_alert_sub_run"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    severity: Mapped[Optional[float]]   = mapped_column(nullable=True)
    max_bloom_prob: Mapped[Optional[float]] = mapped_column(nullable=True)
    max_eri: Mapped[Optional[int]]      = mapped_column(SmallInteger, nullable=True)
    horizon_of_max: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    polygon_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    sent_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    channels_sent: Mapped[Optional[list[str]]] = mapped_column(ARRAY(Text), nullable=True)
    delivery_errors: Mapped[Optional[dict]]    = mapped_column(JSON, nullable=True)

    subscription: Mapped["Subscription"] = relationship(back_populates="alerts")
    run: Mapped["Run"]                   = relationship(back_populates="alerts")
    inapp_notifications: Mapped[list["InAppNotification"]] = relationship(
        back_populates="alert", cascade="all, delete-orphan"
    )


# ──────────────────────────────────────────────────────────────────────────────
# inapp_notifications
# ──────────────────────────────────────────────────────────────────────────────

class InAppNotification(Base):
    __tablename__ = "inapp_notifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False
    )
    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False
    )
    read_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    subscription: Mapped["Subscription"] = relationship(back_populates="inapp_notifications")
    alert: Mapped["Alert"]               = relationship(back_populates="inapp_notifications")
