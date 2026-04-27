"""
app/worker/scheduler.py
-----------------------
APScheduler-based job scheduler.

Jobs:
  - 02:00 UTC daily   → daily_run.run() for yesterday
  - 03:00 UTC daily   → checkpoint_backup.run_backup()

Run as a long-lived process in the `scheduler` Docker service:
    python -m app.worker.scheduler
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import date, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.worker.checkpoint_backup import run_backup
from app.worker.daily_run import run as daily_run

log = logging.getLogger(__name__)


def _daily_job() -> None:
    target = date.today() - timedelta(days=1)
    log.info(f"[scheduler] Starting daily run for {target}")
    try:
        daily_run(target_date=target, triggered_by="scheduler")
    except Exception as exc:
        log.error(f"[scheduler] Daily run failed: {exc}")


def _backup_job() -> None:
    log.info("[scheduler] Starting checkpoint backup")
    try:
        run_backup()
    except Exception as exc:
        log.error(f"[scheduler] Backup failed: {exc}")


def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(_daily_job,  CronTrigger(hour=2,  minute=0), id="daily_run",  misfire_grace_time=3600)
    scheduler.add_job(_backup_job, CronTrigger(hour=3,  minute=0), id="ckpt_backup", misfire_grace_time=3600)

    def _shutdown(sig, frame):
        log.info("[scheduler] Shutting down...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    log.info("[scheduler] Started — daily_run@02:00 UTC, ckpt_backup@03:00 UTC")
    scheduler.start()


if __name__ == "__main__":
    main()
