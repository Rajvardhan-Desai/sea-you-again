"""
/api/alerts/* — public alert history for a subscription.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from app.api.deps import get_session
from app.api.schemas.alert import AlertOut
from app.api.settings import Settings, get_settings
from app.db.models import Alert, Subscription

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/by-subscription/{sub_id}", response_model=list[AlertOut])
def get_alerts_for_subscription(
    sub_id: uuid.UUID,
    token:  str     = Query(...),
    limit:  int     = Query(20, ge=1, le=200),
    db:     Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> list[AlertOut]:
    signer = URLSafeTimedSerializer(settings.secret_key)
    try:
        data       = signer.loads(token, max_age=86400 * 30)
        action     = data.get("action")
        token_sub  = data.get("sub_id")
    except (SignatureExpired, BadSignature):
        raise HTTPException(status_code=403, detail="Invalid or expired token.")

    if action != "unsubscribe" or token_sub != str(sub_id):
        raise HTTPException(status_code=403, detail="Token does not match subscription.")

    alerts = (
        db.query(Alert)
        .filter(Alert.subscription_id == sub_id)
        .order_by(Alert.sent_at.desc().nullslast())
        .limit(limit)
        .all()
    )
    return [AlertOut.model_validate(a) for a in alerts]
