"""
app/worker/notify/inapp.py
--------------------------
Write an in-app notification row. Surfaced on the map page via
GET /api/alerts/by-subscription/{id}?token=... when the subscriber
visits with their signed token in the URL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import InAppNotification

log = logging.getLogger(__name__)


def send_inapp(
    db:              Session,
    subscription_id: uuid.UUID,
    alert_id:        uuid.UUID,
    body:            str,
) -> None:
    notif = InAppNotification(
        subscription_id = subscription_id,
        alert_id        = alert_id,
        body            = body,
    )
    db.add(notif)
    db.flush()   # caller commits
    log.info(f"In-app notification created for sub={subscription_id}")
