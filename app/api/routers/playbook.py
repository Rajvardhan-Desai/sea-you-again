"""
/api/playbook — serve the static YAML playbook as JSON.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from fastapi import APIRouter

router = APIRouter(prefix="/playbook", tags=["playbook"])

_PLAYBOOK_PATH = Path(__file__).resolve().parent.parent.parent.parent / "app" / "playbook" / "playbook.yaml"


@functools.lru_cache(maxsize=1)
def _load_playbook() -> dict:
    with open(_PLAYBOOK_PATH) as f:
        return yaml.safe_load(f)


@router.get("")
def get_playbook() -> dict:
    return _load_playbook()
