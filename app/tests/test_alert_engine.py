"""
test_alert_engine.py
--------------------
Unit tests for the geofenced alert rule.
Uses synthetic bloom/ERI arrays — no model or DB required.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import box


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_npz(path: Path, array: np.ndarray, key: str = "bloom") -> None:
    np.savez_compressed(str(path), **{key: array})


def _make_meta(bbox: list[float]) -> dict:
    return {"bbox": bbox, "horizons": ["2026-04-26"] * 5}


def _make_sub(severity_threshold: float, polygon) -> SimpleNamespace:
    return SimpleNamespace(
        id                 = uuid.uuid4(),
        severity_threshold = severity_threshold,
        geometry           = from_shape(polygon, srid=4326),
        contact_email      = "test@example.com",
        contact_phone      = None,
        channels           = ["inapp"],
        unsubscribe_token  = "tok",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_alert_fires_when_bloom_exceeds_threshold(tmp_path: Path) -> None:
    from app.worker.alert_engine import evaluate_subscription

    # 5 horizons, 4×4 grid; bloom hotspot at pixel (2, 2) = 0.9
    bloom = np.zeros((5, 4, 4), dtype=np.float32)
    bloom[:, 2, 2] = 0.9
    eri   = np.zeros((5, 4, 4), dtype=np.float32)

    _write_npz(tmp_path / "bloom.npz", bloom, "bloom")
    _write_npz(tmp_path / "eri.npz",   eri,   "eri")

    # Polygon covering the full grid  (bbox lon 0-4, lat 0-4)
    bbox  = [0.0, 0.0, 4.0, 4.0]
    poly  = box(0, 0, 4, 4)
    sub   = _make_sub(severity_threshold=0.5, polygon=poly)
    meta  = _make_meta(bbox)

    decision = evaluate_subscription(sub, uuid.uuid4(), tmp_path, meta)

    assert decision.should_fire is True
    assert decision.max_bloom_prob == pytest.approx(0.9, abs=0.01)


def test_alert_suppressed_below_threshold(tmp_path: Path) -> None:
    from app.worker.alert_engine import evaluate_subscription

    bloom = np.full((5, 4, 4), 0.2, dtype=np.float32)
    eri   = np.zeros((5, 4, 4), dtype=np.float32)

    _write_npz(tmp_path / "bloom.npz", bloom, "bloom")
    _write_npz(tmp_path / "eri.npz",   eri,   "eri")

    bbox     = [0.0, 0.0, 4.0, 4.0]
    poly     = box(0, 0, 4, 4)
    sub      = _make_sub(severity_threshold=0.5, polygon=poly)
    meta     = _make_meta(bbox)
    decision = evaluate_subscription(sub, uuid.uuid4(), tmp_path, meta)

    assert decision.should_fire is False


def test_alert_fires_on_eri_class_3(tmp_path: Path) -> None:
    from app.worker.alert_engine import evaluate_subscription

    bloom = np.zeros((5, 4, 4), dtype=np.float32)   # bloom below threshold
    eri   = np.zeros((5, 4, 4), dtype=np.float32)
    eri[:, 1, 1] = 3.0                               # ERI == 3 → fires

    _write_npz(tmp_path / "bloom.npz", bloom, "bloom")
    _write_npz(tmp_path / "eri.npz",   eri,   "eri")

    bbox     = [0.0, 0.0, 4.0, 4.0]
    poly     = box(0, 0, 4, 4)
    sub      = _make_sub(severity_threshold=0.9, polygon=poly)
    meta     = _make_meta(bbox)
    decision = evaluate_subscription(sub, uuid.uuid4(), tmp_path, meta)

    assert decision.should_fire is True
    assert decision.max_eri == 3


def test_no_overlap_produces_no_alert(tmp_path: Path) -> None:
    from app.worker.alert_engine import evaluate_subscription

    bloom = np.full((5, 4, 4), 0.95, dtype=np.float32)
    eri   = np.zeros((5, 4, 4), dtype=np.float32)

    _write_npz(tmp_path / "bloom.npz", bloom, "bloom")
    _write_npz(tmp_path / "eri.npz",   eri,   "eri")

    bbox = [0.0, 0.0, 4.0, 4.0]
    # Polygon far outside the grid
    poly     = box(50, 50, 60, 60)
    sub      = _make_sub(severity_threshold=0.1, polygon=poly)
    meta     = _make_meta(bbox)
    decision = evaluate_subscription(sub, uuid.uuid4(), tmp_path, meta)

    assert decision.should_fire is False
