"""
app/api/deps.py
---------------
FastAPI dependencies — DB session, admin token verification.
"""

from __future__ import annotations

import secrets
from typing import Generator

from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.api.settings import Settings, get_settings
from app.db.session import get_db

# ──────────────────────────────────────────────────────────────────────────────
# Re-export DB dep for convenience
# ──────────────────────────────────────────────────────────────────────────────

def get_session(db: Session = Depends(get_db)) -> Generator[Session, None, None]:
    return db


# ──────────────────────────────────────────────────────────────────────────────
# Admin bearer token
# ──────────────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    mm_admin: str | None = Cookie(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise 401 unless the request carries the ADMIN_TOKEN.

    Accepts either an `Authorization: Bearer <token>` header (used by
    server-to-server callers and the Next.js server-side fetches) or an
    `mm_admin` cookie set by `POST /api/admin/session` (used by browser
    form posts from the admin UI).
    """
    presented = credentials.credentials if credentials else mm_admin
    if presented is None or not secrets.compare_digest(presented, settings.admin_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )
