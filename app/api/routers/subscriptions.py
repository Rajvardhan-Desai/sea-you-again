"""
/api/subscriptions/* — subscription CRUD + magic-link confirm/unsub.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from shapely.geometry import mapping, shape
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.schemas.subscription import SubscriptionCreate, SubscriptionOut, SubscriptionPending
from app.api.settings import Settings, get_settings
from app.db.models import Subscription
from app.worker.notify.email import send_email

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


def _make_token(signer: URLSafeTimedSerializer, sub_id: uuid.UUID, action: str) -> str:
    return signer.dumps({"sub_id": str(sub_id), "action": action})


def _verify_token(
    signer: URLSafeTimedSerializer,
    token: str,
    action: str,
    max_age: int = 86400 * 30,   # 30 days for unsub token
) -> str:
    data = signer.loads(token, max_age=max_age)
    if data.get("action") != action:
        raise ValueError("Wrong token action")
    return data["sub_id"]


@router.post("", response_model=SubscriptionPending, status_code=201)
def create_subscription(
    body:     SubscriptionCreate,
    db:       Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SubscriptionPending:
    signer     = URLSafeTimedSerializer(settings.secret_key)
    sub_id     = uuid.uuid4()
    unsub_tok  = _make_token(signer, sub_id, "unsubscribe")

    # Serialize geometry to WKT for GeoAlchemy2
    geom_wkt = shape(body.geometry.model_dump()).wkt

    sub = Subscription(
        id                 = sub_id,
        name               = body.name,
        contact_email      = str(body.contact_email),
        contact_phone      = body.contact_phone,
        geometry           = f"SRID=4326;{geom_wkt}",
        severity_threshold = body.severity_threshold,
        channels           = body.channels,
        unsubscribe_token  = unsub_tok,
    )
    db.add(sub)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Subscription with this token already exists.")

    # Send confirmation email
    confirm_tok  = _make_token(signer, sub_id, "confirm")
    confirm_url  = f"{settings.public_base_url}/api/subscriptions/confirm?token={confirm_tok}"
    unsub_url    = f"{settings.public_base_url}/api/subscriptions/{sub_id}?token={unsub_tok}"
    send_email(
        to       = str(body.contact_email),
        subject  = "[MM-MARAS] Confirm your bloom alert subscription",
        body     = (
            f"Hello {body.name},\n\n"
            f"Click the link below to confirm your MM-MARAS bloom alert subscription "
            f"for your area of interest:\n\n  {confirm_url}\n\n"
            f"This link expires in 24 hours.\n\n"
            f"To unsubscribe at any time: {unsub_url}\n\n"
            f"— MM-MARAS Automated Early Warning System"
        ),
        settings = settings,
    )

    return SubscriptionPending(id=sub_id)


@router.get("/confirm")
def confirm_subscription(
    token:    str,
    db:       Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    signer = URLSafeTimedSerializer(settings.secret_key)
    try:
        sub_id_str = _verify_token(signer, token, "confirm", max_age=86400)
    except (SignatureExpired, BadSignature, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or expired confirmation token.")

    sub = db.query(Subscription).filter(Subscription.id == uuid.UUID(sub_id_str)).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found.")
    if sub.confirmed_at is None:
        sub.confirmed_at = datetime.now(timezone.utc)
        db.commit()

    return RedirectResponse(url=f"{settings.public_base_url}/subscribe/confirm?ok=1")


@router.get("/{sub_id}", response_model=SubscriptionOut)
def get_subscription(
    sub_id: uuid.UUID,
    token:  str = Query(...),
    db:     Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SubscriptionOut:
    signer = URLSafeTimedSerializer(settings.secret_key)
    try:
        verified_id = _verify_token(signer, token, "unsubscribe")
    except (SignatureExpired, BadSignature, ValueError):
        raise HTTPException(status_code=403, detail="Invalid token.")
    if str(sub_id) != verified_id:
        raise HTTPException(status_code=403, detail="Token does not match subscription.")

    sub = db.query(Subscription).filter(Subscription.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found.")
    return SubscriptionOut.model_validate(sub)


@router.delete("/{sub_id}")
def delete_subscription(
    sub_id: uuid.UUID,
    token:  str = Query(...),
    db:     Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    signer = URLSafeTimedSerializer(settings.secret_key)
    try:
        verified_id = _verify_token(signer, token, "unsubscribe")
    except (SignatureExpired, BadSignature, ValueError):
        raise HTTPException(status_code=403, detail="Invalid token.")
    if str(sub_id) != verified_id:
        raise HTTPException(status_code=403, detail="Token does not match subscription.")

    sub = db.query(Subscription).filter(Subscription.id == sub_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found.")
    db.delete(sub)
    db.commit()
    return {"status": "deleted"}
