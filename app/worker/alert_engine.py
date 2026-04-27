"""
app/worker/alert_engine.py
--------------------------
Geofenced alert rule evaluation.

For each confirmed subscription:
  1. Load bloom.npz + eri.npz from the run artifacts.
  2. Mask the arrays with the subscription's polygon using rasterio.
  3. Compute max_bloom_prob and max_eri across all 5 horizons within the polygon.
  4. Fire an alert if max_bloom_prob > severity_threshold OR max_eri >= 3.
  5. Dispatch email / SMS / in-app notifications.
  6. Upsert an alerts row (UNIQUE on subscription_id, run_id — idempotent).
"""

from __future__ import annotations

import functools
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from shapely.geometry import shape
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.db.models import Alert, Subscription
from app.worker.notify import email as notify_email
from app.worker.notify import inapp as notify_inapp
from app.worker.notify import sms as notify_sms

log = logging.getLogger(__name__)

_PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "playbook" / "playbook.yaml"


@functools.lru_cache(maxsize=1)
def _load_playbook() -> list[dict]:
    with open(_PLAYBOOK_PATH) as f:
        return yaml.safe_load(f)["bands"]


def _severity_band(bloom_prob: float) -> dict:
    for band in reversed(_load_playbook()):
        if bloom_prob >= band["bloom_prob_min"]:
            return band
    return _load_playbook()[0]


def _mask_npz(
    npz_path: Path,
    geom,           # shapely geometry
    bbox: list[float],  # [minlon, minlat, maxlon, maxlat]
    n_horizons: int = 5,
) -> tuple[list[float], int]:
    """
    Return (per-horizon max values, pixel_count) masked by geom.
    Returns ([0]*n_horizons, 0) if no overlap or file missing.
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.mask import mask as rio_mask

    if not npz_path.exists():
        return [0.0] * n_horizons, 0

    arr_data = np.load(npz_path)
    key      = list(arr_data.files)[0]
    arr      = arr_data[key]
    if arr.ndim == 2:
        arr = arr[np.newaxis]  # (1, H, W)

    H, W      = arr.shape[-2], arr.shape[-1]
    minlon, minlat, maxlon, maxlat = bbox
    transform = rasterio.transform.from_bounds(minlon, minlat, maxlon, maxlat, W, H)

    max_vals   = []
    pixel_count = 0

    for h in range(min(n_horizons, arr.shape[0])):
        band    = arr[h].astype(np.float32)
        profile = {
            "driver": "GTiff", "dtype": "float32",
            "width": W, "height": H, "count": 1,
            "crs": "EPSG:4326", "transform": transform,
        }
        with MemoryFile() as mf:
            with mf.open(**profile) as ds:
                ds.write(band, 1)
            with mf.open() as ds:
                try:
                    masked, _ = rio_mask(ds, [geom], crop=True, nodata=np.nan)
                    vals = masked[0][np.isfinite(masked[0])]
                except Exception:
                    vals = np.array([])
        if len(vals) > 0:
            max_vals.append(float(np.nanmax(vals)))
            pixel_count = max(pixel_count, int(len(vals)))
        else:
            max_vals.append(0.0)

    return max_vals, pixel_count


@dataclass
class AlertDecision:
    subscription_id: uuid.UUID
    run_id:          uuid.UUID
    should_fire:     bool
    max_bloom_prob:  float          = 0.0
    max_eri:         int            = 0
    horizon_of_max:  int            = 1
    pixel_count:     int            = 0
    severity:        float          = 0.0
    severity_band:   Optional[dict] = field(default=None)


def evaluate_subscription(
    sub:           Subscription,
    run_id:        uuid.UUID,
    artifacts_path: Path,
    meta:          dict,
) -> AlertDecision:
    """Evaluate one subscription against the run artifacts."""
    bbox = meta.get("bbox", [])
    if not bbox:
        log.warning(f"No bbox in metadata for run {run_id} — skipping sub {sub.id}")
        return AlertDecision(sub.id, run_id, should_fire=False)

    # Deserialise geometry from PostGIS WKB/WKT via shapely
    from geoalchemy2.shape import to_shape
    geom = to_shape(sub.geometry)

    bloom_maxes, pixel_count = _mask_npz(artifacts_path / "bloom.npz", geom, bbox)
    eri_maxes, _             = _mask_npz(artifacts_path / "eri.npz",   geom, bbox)

    max_bloom    = float(np.max(bloom_maxes)) if bloom_maxes else 0.0
    max_eri_val  = int(np.max(eri_maxes))     if eri_maxes  else 0
    h_of_max     = int(np.argmax(bloom_maxes)) + 1 if bloom_maxes else 1

    should_fire = max_bloom > sub.severity_threshold or max_eri_val >= 3

    if pixel_count == 0:
        log.info(f"Sub {sub.id}: polygon has no overlap with ocean mask — no alert.")
        should_fire = False

    return AlertDecision(
        subscription_id = sub.id,
        run_id          = run_id,
        should_fire     = should_fire,
        max_bloom_prob  = max_bloom,
        max_eri         = max_eri_val,
        horizon_of_max  = h_of_max,
        pixel_count     = pixel_count,
        severity        = max_bloom,
        severity_band   = _severity_band(max_bloom) if should_fire else None,
    )


def _build_alert_email_body(
    sub:      Subscription,
    decision: AlertDecision,
    settings,
) -> tuple[str, str]:
    """Return (subject, plain-text body)."""
    band    = decision.severity_band or {}
    actions = band.get("actions", [])
    action_list = "\n".join(f"  • {a}" for a in actions)
    unsub_url   = (
        f"{settings.public_base_url}/api/subscriptions/{sub.id}"
        f"?token={sub.unsubscribe_token}"
    )
    sev_label = band.get("severity", "unknown").upper()

    subject = f"[MM-MARAS] {sev_label} bloom risk — {sub.name}"
    body    = (
        f"Hello {sub.name},\n\n"
        f"MM-MARAS has detected a {sev_label} bloom risk in your subscribed area.\n\n"
        f"  Max bloom probability : {decision.max_bloom_prob:.0%}\n"
        f"  Max ERI class         : {decision.max_eri} / 4\n"
        f"  Peak horizon          : Day +{decision.horizon_of_max}\n\n"
        f"Recommended actions:\n{action_list}\n\n"
        f"View the latest forecast map:\n  {settings.public_base_url}/map\n\n"
        f"Unsubscribe: {unsub_url}\n\n"
        f"— MM-MARAS Automated Early Warning System\n"
        f"  Bay of Bengal Chlorophyll-a Forecast · v3.6"
    )
    return subject, body


def _build_sms_body(decision: AlertDecision, settings) -> str:
    band  = decision.severity_band or {}
    sev   = band.get("severity", "?").upper()
    return (
        f"[MM-MARAS] {sev} bloom risk. "
        f"Max prob {decision.max_bloom_prob:.0%} on Day+{decision.horizon_of_max}. "
        f"Check {settings.public_base_url}/map for details."
    )


def _upsert_alert(
    db:       Session,
    decision: AlertDecision,
    channels: list[str],
    errors:   dict,
) -> Alert:
    """Insert or update the alerts row (idempotent via UNIQUE constraint)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    alert = (
        db.query(Alert)
        .filter(
            Alert.subscription_id == decision.subscription_id,
            Alert.run_id          == decision.run_id,
        )
        .first()
    )
    if alert is None:
        alert = Alert(
            subscription_id = decision.subscription_id,
            run_id          = decision.run_id,
            severity        = decision.severity,
            max_bloom_prob  = decision.max_bloom_prob,
            max_eri         = decision.max_eri,
            horizon_of_max  = decision.horizon_of_max,
            polygon_summary = {"pixel_count": decision.pixel_count},
            channels_sent   = channels,
            delivery_errors = errors if errors else None,
            sent_at         = datetime.now(timezone.utc) if channels else None,
        )
        db.add(alert)
    else:
        if channels:
            alert.channels_sent   = list(set((alert.channels_sent or []) + channels))
            alert.delivery_errors = errors if errors else alert.delivery_errors
            alert.sent_at         = alert.sent_at or datetime.now(timezone.utc)
    db.flush()
    return alert


def run_alert_evaluation(
    db:             Session,
    run_id:         uuid.UUID,
    artifacts_path: Path,
    meta:           dict,
    settings,
) -> dict:
    """
    Evaluate all confirmed subscriptions and dispatch notifications.
    Returns summary dict for inference_metrics / logging.
    """
    subs = db.query(Subscription).filter(Subscription.confirmed_at.isnot(None)).all()
    log.info(f"Alert evaluation: {len(subs)} confirmed subscriptions")

    fired     = 0
    skipped   = 0
    n_errors  = 0

    for sub in subs:
        decision = evaluate_subscription(sub, run_id, artifacts_path, meta)

        if not decision.should_fire:
            skipped += 1
            continue

        fired      += 1
        channels_ok = []
        errors:      dict = {}

        subject, body = _build_alert_email_body(sub, decision, settings)

        if "email" in sub.channels:
            try:
                _dispatch_email(sub.contact_email, subject, body, settings)
                channels_ok.append("email")
            except Exception as exc:
                log.warning(f"Email failed for sub {sub.id}: {exc}")
                errors["email"] = str(exc)
                n_errors += 1

        if "sms" in sub.channels and sub.contact_phone:
            try:
                _dispatch_sms(sub.contact_phone, _build_sms_body(decision, settings), settings)
                channels_ok.append("sms")
            except Exception as exc:
                log.warning(f"SMS failed for sub {sub.id}: {exc}")
                errors["sms"] = str(exc)
                n_errors += 1

        alert = _upsert_alert(db, decision, channels_ok, errors)

        if "inapp" in sub.channels:
            try:
                notify_inapp.send_inapp(db, sub.id, alert.id, body[:500])
                if "inapp" not in (alert.channels_sent or []):
                    alert.channels_sent = (alert.channels_sent or []) + ["inapp"]
            except Exception as exc:
                log.warning(f"In-app notify failed for sub {sub.id}: {exc}")
                n_errors += 1

    db.commit()
    log.info(f"Alerts: fired={fired} skipped={skipped} errors={n_errors}")
    return {"alerts_fired": fired, "alerts_skipped": skipped, "notify_errors": n_errors}


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=30))
def _dispatch_email(to: str, subject: str, body: str, settings) -> None:
    notify_email.send_email(to, subject, body, settings)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=30))
def _dispatch_sms(to: str, body: str, settings) -> None:
    notify_sms.send_sms(to, body, settings)
