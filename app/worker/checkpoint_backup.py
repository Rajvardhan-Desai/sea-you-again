"""
app/worker/checkpoint_backup.py
--------------------------------
Copy model checkpoints to a backup directory, retaining last 14 dated copies.

Scheduled by APScheduler at 03:00 UTC.
Can also be triggered via POST /api/admin/checkpoint-backup.

Usage:
    python -m app.worker.checkpoint_backup
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

_REPO        = Path(__file__).resolve().parent.parent.parent
_CKPT_DIR    = _REPO / "model" / "checkpoints"
_RETAIN_DAYS = 14


def run_backup(
    checkpoint_dir: Path | None = None,
    backup_dir:     Path | None = None,
) -> dict:
    """
    Copy all *.pt files from checkpoint_dir → backup_dir/<today>/.
    Prune dated subdirs beyond RETAIN_DAYS.
    Returns a summary dict.
    """
    ckpt_dir = checkpoint_dir or _CKPT_DIR
    bak_root = backup_dir or Path(os.environ.get("CHECKPOINT_BACKUP_DIR", "/data/backups"))

    today     = date.today()
    bak_dir   = bak_root / str(today)
    bak_dir.mkdir(parents=True, exist_ok=True)

    pts      = list(ckpt_dir.glob("*.pt"))
    copied   = []

    for src in pts:
        dst = bak_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
        log.info(f"Backed up {src.name} → {dst}")

    # Prune old backups
    dated_dirs = sorted(
        (d for d in bak_root.iterdir() if d.is_dir()),
        key=lambda d: d.name,
    )
    pruned = []
    while len(dated_dirs) > _RETAIN_DAYS:
        oldest = dated_dirs.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)
        pruned.append(oldest.name)
        log.info(f"Pruned old backup: {oldest}")

    result = {
        "date":    str(today),
        "copied":  copied,
        "pruned":  pruned,
    }
    log.info(f"Checkpoint backup complete: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_backup()
