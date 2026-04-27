"""
app/db/session.py
-----------------
SQLAlchemy 2.x engine + sessionmaker.
Import `SessionLocal` everywhere; use as a context manager or via FastAPI deps.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/mmaras",
)

engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a DB session and close it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
