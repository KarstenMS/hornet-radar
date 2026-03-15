"""Hornet Radar: cleanup of old events from local storage based on age and total size limits."""

import os
import logging
import time
import shutil
from config import EVENTS_DIR, EVENT_RETENTION_DAYS, MAX_EVENT_STORAGE_GB

logger = logging.getLogger(__name__)

def _get_size(path):
    """Return total size of file or directory."""
    if os.path.isfile(path):
        return os.path.getsize(path)

    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def cleanup_events():
    """
    Cleanup event directory:
    1. Delete events older than EVENT_RETENTION_DAYS
    2. Ensure directory stays under MAX_EVENT_STORAGE_GB
    """
    logger.info("[Cleanup] Starting cleanup of old events")

    if not os.path.exists(EVENTS_DIR):
        return

    now = time.time()
    cutoff = now - (EVENT_RETENTION_DAYS * 86400)
    max_bytes = MAX_EVENT_STORAGE_GB * 1024**3

    entries = []

    # --- Scan directory ---
    for name in os.listdir(EVENTS_DIR):

        path = os.path.join(EVENTS_DIR, name)

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue

        size = _get_size(path)

        entries.append({
            "path": path,
            "mtime": mtime,
            "size": size
        })

    deleted = 0

    # --- Step 1: delete by age ---
    for e in entries:
        if e["mtime"] < cutoff:
            try:
                if os.path.isdir(e["path"]):
                    shutil.rmtree(e["path"])
                else:
                    os.remove(e["path"])
                deleted += 1
            except Exception as err:
                logger.info("[Cleanup] Failed deleting {e['path']}: {err}")
               

    # refresh list
    entries = [e for e in entries if os.path.exists(e["path"])]

    # --- Step 2: enforce storage limit ---
    total_size = sum(e["size"] for e in entries)

    if total_size > max_bytes:

        # oldest first
        entries.sort(key=lambda e: e["mtime"])

        for e in entries:

            if total_size <= max_bytes:
                break

            try:
                if os.path.isdir(e["path"]):
                    shutil.rmtree(e["path"])
                else:
                    os.remove(e["path"])

                total_size -= e["size"]
                deleted += 1

            except Exception as err:
                logger.info("[Cleanup] Failed deleting {e['path']}: {err}")

    if deleted > 0:
        logger.info("[Cleanup] Removed {deleted} old events")
    else:
        logger.info("[Cleanup] No old events to remove")