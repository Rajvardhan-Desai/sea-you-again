"""
app/worker/trigger_listener.py
------------------------------
Long-lived BLPOP loop that consumes ad-hoc job requests pushed by the api
container into Redis. Lives inside the scheduler container, alongside the
APScheduler cron jobs, because that container has the heavy ML deps,
data-preprocessing-pipeline source, and the model checkpoint mount.

Queues (LPUSH'd by app/api/routers/admin.py):
  - mmaras:trigger_run        {"run_date": "...", "triggered_by": "..."}
  - mmaras:retry_run          {"run_id":   "...", "run_date": "...",
                               "phase":    "...", "triggered_by": "..."}
  - mmaras:checkpoint_backup  {"triggered_by": "..."}

Jobs are processed serially — daily_run is heavy and the rest of the system
assumes one run at a time.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date

import redis

log = logging.getLogger(__name__)

QUEUES = (
    "mmaras:trigger_run",
    "mmaras:retry_run",
    "mmaras:checkpoint_backup",
)


def _handle_trigger(payload: dict) -> None:
    from app.worker.daily_run import run as daily_run

    target = date.fromisoformat(payload["run_date"])
    log.info("[listener] daily_run target=%s triggered_by=%s",
             target, payload.get("triggered_by"))
    daily_run(target_date=target, triggered_by=payload.get("triggered_by", "admin:api"))


def _handle_retry(payload: dict) -> None:
    from app.worker.daily_run import run as daily_run

    target = date.fromisoformat(payload["run_date"])
    phase  = payload.get("phase", "ingest")
    log.info("[listener] retry target=%s phase=%s triggered_by=%s",
             target, phase, payload.get("triggered_by"))
    daily_run(
        target_date  = target,
        triggered_by = payload.get("triggered_by", "admin:retry"),
        start_phase  = phase,
    )


def _handle_backup(payload: dict) -> None:
    from app.worker.checkpoint_backup import run_backup

    log.info("[listener] checkpoint_backup triggered_by=%s",
             payload.get("triggered_by"))
    run_backup()


_DISPATCH = {
    "mmaras:trigger_run":       _handle_trigger,
    "mmaras:retry_run":         _handle_retry,
    "mmaras:checkpoint_backup": _handle_backup,
}


def listen(redis_url: str | None = None) -> None:
    """Block forever, consuming jobs from QUEUES. Safe to run in a thread."""
    url    = redis_url or os.environ.get("REDIS_URL", "redis://redis:6379/0")
    client = redis.Redis.from_url(url, decode_responses=True)
    log.info("[listener] connected to %s; queues=%s", url, list(QUEUES))

    while True:
        try:
            item = client.blpop(QUEUES, timeout=5)
        except redis.exceptions.RedisError as exc:
            log.error("[listener] redis error: %s — retrying in 5s", exc)
            import time
            time.sleep(5)
            continue
        if item is None:
            continue
        queue_name, raw = item
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.error("[listener] bad payload on %s: %r", queue_name, raw)
            continue
        handler = _DISPATCH.get(queue_name)
        if handler is None:
            log.warning("[listener] no handler for %s", queue_name)
            continue
        try:
            handler(payload)
        except Exception:
            log.exception("[listener] job failed on %s payload=%s",
                          queue_name, payload)


if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    listen()
