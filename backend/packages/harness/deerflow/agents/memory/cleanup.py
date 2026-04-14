"""Scheduled memory cleanup — periodically clears per-user and (optionally) global memory."""

import logging
import threading
from typing import Any

from deerflow.agents.memory.storage import create_empty_memory, get_memory_storage
from deerflow.config.memory_config import get_memory_config

logger = logging.getLogger(__name__)


class MemoryCleanupScheduler:
    """Background scheduler that periodically clears memory files.

    The scheduler respects the following ``MemoryCleanupConfig`` knobs:

    * ``enabled`` — master switch; ``start()`` is a no-op when ``False``.
    * ``interval_hours`` — how often cleanup runs (in hours).
    * ``apply_to_global`` — whether the shared ``memory/global.json`` is also
      cleared on each cycle.

    Per-user memory for every channel that has ``per_user`` enabled in
    ``MemoryConfig`` is always cleared when cleanup runs.  The scheduler
    enumerates all ``*.json`` files under each channel's memory directory and
    resets them to the empty structure.
    """

    def __init__(self) -> None:
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the cleanup scheduler if enabled in config."""
        config = get_memory_config()
        if not config.cleanup.enabled:
            logger.info("Memory cleanup scheduler is disabled")
            return

        with self._lock:
            if self._running:
                return
            self._running = True

        self._schedule_next()
        logger.info(
            "Memory cleanup scheduler started (interval=%dh, apply_to_global=%s)",
            config.cleanup.interval_hours,
            config.cleanup.apply_to_global,
        )

    def stop(self) -> None:
        """Stop the cleanup scheduler."""
        with self._lock:
            self._running = False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        logger.info("Memory cleanup scheduler stopped")

    def _schedule_next(self) -> None:
        """Schedule the next cleanup run."""
        config = get_memory_config()
        interval_seconds = config.cleanup.interval_hours * 3600

        with self._lock:
            if not self._running:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(interval_seconds, self._run_cleanup)
            self._timer.daemon = True
            self._timer.start()

    def _run_cleanup(self) -> None:
        """Execute a cleanup cycle: clear per-user memory and optionally global."""
        config = get_memory_config()
        storage = get_memory_storage()

        logger.info("Memory cleanup cycle started")
        cleaned_count = 0

        # Clear per-user memory for each channel that has per_user enabled
        per_user_cfg = config.per_user
        for channel_name in ("feishu",):
            if not getattr(per_user_cfg, channel_name, False):
                continue
            try:
                user_files = storage.list_user_memory_files(channel_name)
                for user_file in user_files:
                    try:
                        # Extract user_id from filename (e.g. "ou_xxx.json" → "ou_xxx")
                        user_id = user_file.stem
                        storage.clear_user_memory(channel_name, user_id)
                        cleaned_count += 1
                    except Exception:
                        logger.exception("Failed to clear user memory: %s", user_file)
            except Exception:
                logger.exception("Failed to list user memory files for channel %s", channel_name)

        # Optionally clear global memory
        if config.cleanup.apply_to_global:
            try:
                storage.save(create_empty_memory())
                cleaned_count += 1
                logger.info("Global memory cleared by cleanup scheduler")
            except Exception:
                logger.exception("Failed to clear global memory during cleanup")

        logger.info("Memory cleanup cycle completed: %d file(s) cleared", cleaned_count)

        # Schedule the next cycle
        if self._running:
            self._schedule_next()

    def run_cleanup_now(self) -> dict[str, Any]:
        """Run cleanup immediately (for manual/API triggering).

        Returns a summary dict with the number of files cleaned.
        """
        config = get_memory_config()
        if not config.cleanup.enabled:
            return {"status": "disabled", "cleaned": 0}

        self._run_cleanup()
        return {"status": "ok"}


# -- Singleton ---------------------------------------------------------------

_cleanup_scheduler: MemoryCleanupScheduler | None = None
_scheduler_lock = threading.Lock()


def get_cleanup_scheduler() -> MemoryCleanupScheduler:
    """Get or create the global cleanup scheduler singleton."""
    global _cleanup_scheduler
    if _cleanup_scheduler is not None:
        return _cleanup_scheduler

    with _scheduler_lock:
        if _cleanup_scheduler is not None:
            return _cleanup_scheduler
        _cleanup_scheduler = MemoryCleanupScheduler()

    return _cleanup_scheduler


def start_cleanup_scheduler() -> None:
    """Convenience function to start the global cleanup scheduler."""
    get_cleanup_scheduler().start()


def stop_cleanup_scheduler() -> None:
    """Convenience function to stop the global cleanup scheduler."""
    scheduler = get_cleanup_scheduler()
    scheduler.stop()
