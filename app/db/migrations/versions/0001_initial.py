"""Initial schema — runs, data_sources, subscriptions, alerts, inapp_notifications.

Revision ID: 0001
Revises:
Create Date: 2026-04-26 00:00:00
"""

from __future__ import annotations

import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    # Enums
    run_status  = postgresql.ENUM(
        "pending", "ingesting", "inferring", "alerting",
        "succeeded", "failed", "partial",
        name="run_status",
    )
    src_status  = postgresql.ENUM("ok", "retrying", "failed", name="source_status")
    src_name    = postgresql.ENUM("cmems_chl", "cmems_phys", "era5", "glofas", name="source_name")
    run_status.create(op.get_bind(), checkfirst=True)
    src_status.create(op.get_bind(), checkfirst=True)
    src_name.create(op.get_bind(), checkfirst=True)

    # runs
    op.create_table(
        "runs",
        sa.Column("id",                postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("run_date",          sa.Date,   nullable=False, unique=True),
        sa.Column("status",            run_status, nullable=False, server_default="pending"),
        sa.Column("started_at",        sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at",       sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("ingest_summary",    postgresql.JSON, nullable=True),
        sa.Column("inference_metrics", postgresql.JSON, nullable=True),
        sa.Column("artifacts_path",    sa.Text, nullable=True),
        sa.Column("error_text",        sa.Text, nullable=True),
        sa.Column("triggered_by",      sa.String(64), nullable=True),
    )

    # data_sources
    op.create_table(
        "data_sources",
        sa.Column("id",               postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("run_id",           postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source",           src_name, nullable=False),
        sa.Column("status",           src_status, nullable=False, server_default="ok"),
        sa.Column("attempts",         sa.Integer, nullable=False, server_default="1"),
        sa.Column("bytes_written",    sa.BigInteger, nullable=True),
        sa.Column("remote_latest_ts", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("message",          sa.Text, nullable=True),
    )

    # subscriptions  (contact_email uses CITEXT for case-insensitive uniqueness)
    op.create_table(
        "subscriptions",
        sa.Column("id",                 postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("name",               sa.Text, nullable=False),
        sa.Column("contact_email",      sa.Text, nullable=False),   # would be CITEXT but SA maps to Text
        sa.Column("contact_phone",      sa.Text, nullable=True),
        sa.Column("geometry",           Geography(geometry_type="POLYGON", srid=4326), nullable=False),
        sa.Column("severity_threshold", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("channels",           postgresql.ARRAY(sa.Text), nullable=False),
        sa.Column("confirmed_at",       sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("unsubscribe_token",  sa.Text, nullable=False, unique=True),
        sa.Column("created_at",         sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("severity_threshold >= 0 AND severity_threshold <= 1",
                           name="ck_severity_range"),
    )
    op.create_index(
        "ix_subscriptions_geometry", "subscriptions", ["geometry"], postgresql_using="gist"
    )

    # alerts
    op.create_table(
        "alerts",
        sa.Column("id",               postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("subscription_id",  postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id",           postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("severity",         sa.Float,       nullable=True),
        sa.Column("max_bloom_prob",   sa.Float,       nullable=True),
        sa.Column("max_eri",          sa.SmallInteger, nullable=True),
        sa.Column("horizon_of_max",   sa.SmallInteger, nullable=True),
        sa.Column("polygon_summary",  postgresql.JSON, nullable=True),
        sa.Column("sent_at",          sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("channels_sent",    postgresql.ARRAY(sa.Text), nullable=True),
        sa.Column("delivery_errors",  postgresql.JSON, nullable=True),
        sa.UniqueConstraint("subscription_id", "run_id", name="uq_alert_sub_run"),
    )

    # inapp_notifications
    op.create_table(
        "inapp_notifications",
        sa.Column("id",              postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("subscription_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("alert_id",        postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("read_at",         sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("body",            sa.Text, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("inapp_notifications")
    op.drop_table("alerts")
    op.drop_index("ix_subscriptions_geometry", table_name="subscriptions")
    op.drop_table("subscriptions")
    op.drop_table("data_sources")
    op.drop_table("runs")
