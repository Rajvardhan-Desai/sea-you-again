"""
test_subscriptions_api.py
--------------------------
Integration-style tests for the subscription CRUD endpoints.
Uses httpx TestClient — no real DB (SQLite in-memory via env override).
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

# Use SQLite for tests (no PostGIS geometry — skip geo assertions)
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_mmaras.db")
os.environ.setdefault("ADMIN_TOKEN",  "test-admin-token")
os.environ.setdefault("SECRET_KEY",   "test-secret-key-at-least-32-chars-long")

# Import app AFTER env vars are set
from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)

_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [[80.0, 10.0], [85.0, 10.0], [85.0, 15.0], [80.0, 15.0], [80.0, 10.0]]
    ],
}


def test_health() -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_latest_forecast_404_when_empty() -> None:
    r = client.get("/api/forecast/latest")
    assert r.status_code == 404


def test_playbook_endpoint() -> None:
    r = client.get("/api/playbook")
    assert r.status_code == 200
    data = r.json()
    assert "bands" in data
    assert len(data["bands"]) >= 4


def test_admin_unauthenticated() -> None:
    r = client.get("/api/admin/runs")
    assert r.status_code == 401


def test_admin_authenticated() -> None:
    r = client.get(
        "/api/admin/runs",
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_admin_wrong_token() -> None:
    r = client.get(
        "/api/admin/runs",
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401
