"""
/api/admin/* — protected admin endpoints (require Bearer ADMIN_TOKEN).
"""

from __future__ import annotations

import secrets
import subprocess
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_session, require_admin
from app.api.schemas.admin import (
    AdminSessionRequest,
    DataSourceOut,
    RunSummary,
    TriggerRunRequest,
)
from app.api.settings import Settings, get_settings
from app.db.models import Alert, DataSource, Run, Subscription

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


# ──────────────────────────────────────────────────────────────────────────────
# Session (cookie exchange — no require_admin dep so we can check inline)
# ──────────────────────────────────────────────────────────────────────────────

_session_router = APIRouter(prefix="/admin", tags=["admin"])

@_session_router.post("/session")
def create_admin_session(
    body:     AdminSessionRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> dict:
    if not secrets.compare_digest(body.token, settings.admin_token):
        raise HTTPException(status_code=401, detail="Invalid token")
    response.set_cookie(
        "mm_admin", body.token,
        httponly=True, samesite="strict", max_age=86400 * 7,
    )
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Runs
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    limit: int     = Query(50, ge=1, le=500),
    db:    Session = Depends(get_session),
) -> list[RunSummary]:
    runs = db.query(Run).order_by(desc(Run.run_date)).limit(limit).all()
    return [RunSummary.model_validate(r) for r in runs]


@router.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_session)) -> RunSummary:
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return RunSummary.model_validate(run)


@router.post("/runs/trigger", response_model=RunSummary, status_code=202)
def trigger_run(
    body: TriggerRunRequest,
    db:   Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> RunSummary:
    """Enqueue a manual daily run by spawning daily_run as a subprocess."""
    target_date = body.run_date or (date.today() - timedelta(days=1))

    # Check if a run already exists for this date
    existing = db.query(Run).filter(Run.run_date == target_date).first()
    if existing and existing.status in ("ingesting", "inferring", "alerting"):
        raise HTTPException(status_code=409, detail=f"Run for {target_date} already in progress.")

    # Spawn as background subprocess — worker container handles it
    subprocess.Popen(
        ["python", "-m", "app.worker.daily_run", "--date", str(target_date),
         "--triggered-by", "admin:api"],
        start_new_session=True,
    )

    # Return stub (the subprocess will upsert the DB row)
    if existing:
        return RunSummary.model_validate(existing)
    stub = Run(
        run_date    = target_date,
        status      = "pending",
        triggered_by = "admin:api",
    )
    db.add(stub)
    db.commit()
    db.refresh(stub)
    return RunSummary.model_validate(stub)


@router.post("/runs/{run_id}/retry")
def retry_run(
    run_id: uuid.UUID,
    phase:  str = Query("ingest", pattern="^(ingest|infer|alert)$"),
    db:     Session = Depends(get_session),
) -> dict:
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    subprocess.Popen(
        ["python", "-m", "app.worker.daily_run",
         "--date", str(run.run_date),
         "--triggered-by", "admin:retry",
         "--start-phase", phase],
        start_new_session=True,
    )
    return {"status": "retry_dispatched", "run_id": str(run_id), "phase": phase}


# ──────────────────────────────────────────────────────────────────────────────
# Data sources
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/data-sources", response_model=list[DataSourceOut])
def list_data_sources(
    limit: int     = Query(100, ge=1, le=1000),
    db:    Session = Depends(get_session),
) -> list[DataSourceOut]:
    sources = db.query(DataSource).order_by(desc(DataSource.run_id)).limit(limit).all()
    return [DataSourceOut.model_validate(s) for s in sources]


# ──────────────────────────────────────────────────────────────────────────────
# Subscriptions
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/subscriptions")
def list_subscriptions(
    limit:  int  = Query(100, ge=1, le=1000),
    offset: int  = Query(0, ge=0),
    db:     Session = Depends(get_session),
) -> dict:
    total = db.query(Subscription).count()
    subs  = db.query(Subscription).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id":                str(s.id),
                "name":              s.name,
                "contact_email":     s.contact_email,
                "channels":          s.channels,
                "severity_threshold": s.severity_threshold,
                "confirmed_at":      s.confirmed_at.isoformat() if s.confirmed_at else None,
                "created_at":        s.created_at.isoformat(),
            }
            for s in subs
        ],
    }


@router.delete("/subscriptions/{sub_id}")
def admin_delete_subscription(
    sub_id: uuid.UUID,
    db:     Session = Depends(get_session),
) -> dict:
    sub = db.query(Subscription).filter(Subscription.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found.")
    db.delete(sub)
    db.commit()
    return {"status": "deleted"}


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint backup
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/checkpoint-backup")
def trigger_checkpoint_backup() -> dict:
    subprocess.Popen(
        ["python", "-m", "app.worker.checkpoint_backup"],
        start_new_session=True,
    )
    return {"status": "backup_dispatched"}


# Export session router separately (no auth dep)
def include_session_router(app) -> None:
    app.include_router(_session_router)
