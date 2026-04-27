"""
app/api/deps.py
---------------
FastAPI dependencies — DB session, admin token verification.
"""

from __future__ import annotations

import secrets
from typing import Generator

from fastapi import Depends, HTTPException, status
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
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise 401 if the bearer token does not match ADMIN_TOKEN."""
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, settings.admin_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )
