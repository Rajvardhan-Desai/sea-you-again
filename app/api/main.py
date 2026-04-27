"""
app/api/main.py
---------------
FastAPI application factory.

Startup sequence:
  1. Run alembic migrations (safe to run on every start — idempotent).
  2. Mount Prometheus /metrics endpoint.
  3. Register all routers.
  4. Configure CORS.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app.api.routers import admin as admin_router_module
from app.api.routers.admin import router as admin_router
from app.api.routers.alerts import router as alerts_router
from app.api.routers.forecast import router as forecast_router
from app.api.routers.playbook import router as playbook_router
from app.api.routers.subscriptions import router as subscriptions_router
from app.api.settings import get_settings

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent


def _run_migrations() -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            log.warning(f"Alembic migration warning:\n{result.stderr}")
        else:
            log.info("DB migrations: up to date.")
    except Exception as exc:
        log.error(f"Alembic migration failed: {exc}")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title       = "MM-MARAS Dashboard API",
        version     = settings.version,
        description = "Bay of Bengal Chl-a early-warning system — forecast, bloom alerts, ERI.",
        docs_url    = "/api/docs",
        redoc_url   = "/api/redoc",
        openapi_url = "/api/openapi.json",
    )

    # CORS — allow the Next.js frontend and any configured origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = [settings.public_base_url, "http://localhost:3000"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Routers
    prefix = "/api"
    app.include_router(forecast_router,      prefix=prefix)
    app.include_router(subscriptions_router, prefix=prefix)
    app.include_router(alerts_router,        prefix=prefix)
    app.include_router(playbook_router,      prefix=prefix)
    app.include_router(admin_router,         prefix=prefix)
    # Session endpoint (no auth dep — checked inline)
    admin_router_module.include_session_router(
        type("_app", (), {"include_router": lambda self, r: app.include_router(r, prefix=prefix)})()
    )

    @app.get("/api/health", tags=["health"])
    def health() -> dict:
        from app.db.session import engine
        from sqlalchemy import text
        from app.db.models import Run
        from sqlalchemy.orm import Session

        with Session(engine) as db:
            try:
                latest = db.query(Run).filter(Run.status == "succeeded").order_by(Run.run_date.desc()).first()
                latest_date = str(latest.run_date) if latest else None
            except Exception:
                latest_date = None

        return {
            "status":           "ok",
            "version":          settings.version,
            "latest_run_date":  latest_date,
        }

    @app.on_event("startup")
    async def on_startup() -> None:
        _run_migrations()

    return app


app = create_app()
