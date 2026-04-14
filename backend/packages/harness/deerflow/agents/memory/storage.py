"""Memory storage providers."""

import abc
import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from deerflow.config.agents_config import AGENT_NAME_PATTERN
from deerflow.config.memory_config import get_memory_config
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)


def create_empty_memory() -> dict[str, Any]:
    """Create an empty memory structure."""
    return {
        "version": "1.0",
        "lastUpdated": datetime.utcnow().isoformat() + "Z",
        "user": {
            "workContext": {"summary": "", "updatedAt": ""},
            "personalContext": {"summary": "", "updatedAt": ""},
            "topOfMind": {"summary": "", "updatedAt": ""},
        },
        "history": {
            "recentMonths": {"summary": "", "updatedAt": ""},
            "earlierContext": {"summary": "", "updatedAt": ""},
            "longTermBackground": {"summary": "", "updatedAt": ""},
        },
        "facts": [],
    }


class MemoryStorage(abc.ABC):
    """Abstract base class for memory storage providers."""

    @abc.abstractmethod
    def load(self, agent_name: str | None = None) -> dict[str, Any]:
        """Load memory data for the given agent."""
        pass

    @abc.abstractmethod
    def reload(self, agent_name: str | None = None) -> dict[str, Any]:
        """Force reload memory data for the given agent."""
        pass

    @abc.abstractmethod
    def save(self, memory_data: dict[str, Any], agent_name: str | None = None) -> bool:
        """Save memory data for the given agent."""
        pass

    # -- Per-user memory operations -----------------------------------------

    def load_user_memory(self, channel_name: str, user_id: str) -> dict[str, Any]:
        """Load per-user memory data.

        Default implementation delegates to file-based storage.
        Subclasses may override for custom backends.
        """
        raise NotImplementedError

    def reload_user_memory(self, channel_name: str, user_id: str) -> dict[str, Any]:
        """Force-reload per-user memory data."""
        raise NotImplementedError

    def save_user_memory(self, memory_data: dict[str, Any], channel_name: str, user_id: str) -> bool:
        """Save per-user memory data."""
        raise NotImplementedError

    def clear_user_memory(self, channel_name: str, user_id: str) -> bool:
        """Clear per-user memory (reset to empty)."""
        raise NotImplementedError

    def list_user_memory_files(self, channel_name: str) -> list[Path]:
        """List all per-user memory files for a channel."""
        raise NotImplementedError


# Cache key type: None = global, str = agent_name, tuple = (channel, user_id)
_CacheKey = str | tuple[str, str] | None


class FileMemoryStorage(MemoryStorage):
    """File-based memory storage provider."""

    def __init__(self):
        """Initialize the file memory storage."""
        # Per-agent/user memory cache: keyed by cache key
        # Value: (memory_data, file_mtime)
        self._memory_cache: dict[_CacheKey, tuple[dict[str, Any], float | None]] = {}
        self._migration_done = False

    def _validate_agent_name(self, agent_name: str) -> None:
        """Validate that the agent name is safe to use in filesystem paths.

        Uses the repository's established AGENT_NAME_PATTERN to ensure consistency
        across the codebase and prevent path traversal or other problematic characters.
        """
        if not agent_name:
            raise ValueError("Agent name must be a non-empty string.")
        if not AGENT_NAME_PATTERN.match(agent_name):
            raise ValueError(f"Invalid agent name {agent_name!r}: names must match {AGENT_NAME_PATTERN.pattern}")

    def _maybe_migrate_legacy(self) -> None:
        """Migrate legacy ``memory.json`` to ``memory/global.json`` if needed.

        Called once lazily on the first global-memory access.  The old file is
        *copied* (not moved) so that downgrade scenarios don't lose data.
        """
        if self._migration_done:
            return
        self._migration_done = True

        paths = get_paths()
        legacy = paths.memory_file  # .deer-flow/memory.json
        target = paths.global_memory_file  # .deer-flow/memory/global.json

        if legacy.exists() and not target.exists():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(legacy), str(target))
                logger.info("Migrated legacy memory.json → memory/global.json")
            except OSError as e:
                logger.warning("Failed to migrate legacy memory.json: %s", e)

    def _get_memory_file_path(self, agent_name: str | None = None) -> Path:
        """Get the path to the memory file."""
        if agent_name is not None:
            self._validate_agent_name(agent_name)
            return get_paths().agent_memory_file(agent_name)

        config = get_memory_config()
        if config.storage_path:
            p = Path(config.storage_path)
            return p if p.is_absolute() else get_paths().base_dir / p

        # Default: use the new memory/global.json with legacy migration
        self._maybe_migrate_legacy()
        return get_paths().global_memory_file

    def _get_user_memory_file_path(self, channel_name: str, user_id: str) -> Path:
        """Get the path to a per-user memory file."""
        return get_paths().user_memory_file(channel_name, user_id)

    def _load_memory_from_file(self, file_path: Path) -> dict[str, Any]:
        """Load memory data from a specific file path."""
        if not file_path.exists():
            return create_empty_memory()

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load memory file: %s", e)
            return create_empty_memory()

    def _load_cached(self, cache_key: _CacheKey, file_path: Path) -> dict[str, Any]:
        """Load memory data with mtime-based caching."""
        try:
            current_mtime = file_path.stat().st_mtime if file_path.exists() else None
        except OSError:
            current_mtime = None

        cached = self._memory_cache.get(cache_key)
        if cached is None or cached[1] != current_mtime:
            memory_data = self._load_memory_from_file(file_path)
            self._memory_cache[cache_key] = (memory_data, current_mtime)
            return memory_data

        return cached[0]

    def _save_to_file(self, memory_data: dict[str, Any], file_path: Path, cache_key: _CacheKey) -> bool:
        """Save memory data to a file and update cache."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            memory_data["lastUpdated"] = datetime.utcnow().isoformat() + "Z"

            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)

            temp_path.replace(file_path)

            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = None

            self._memory_cache[cache_key] = (memory_data, mtime)
            logger.info("Memory saved to %s", file_path)
            return True
        except OSError as e:
            logger.error("Failed to save memory file: %s", e)
            return False

    # -- MemoryStorage interface (agent / global) ----------------------------

    def load(self, agent_name: str | None = None) -> dict[str, Any]:
        """Load memory data (cached with file modification time check)."""
        file_path = self._get_memory_file_path(agent_name)
        return self._load_cached(agent_name, file_path)

    def reload(self, agent_name: str | None = None) -> dict[str, Any]:
        """Reload memory data from file, forcing cache invalidation."""
        file_path = self._get_memory_file_path(agent_name)
        memory_data = self._load_memory_from_file(file_path)

        try:
            mtime = file_path.stat().st_mtime if file_path.exists() else None
        except OSError:
            mtime = None

        self._memory_cache[agent_name] = (memory_data, mtime)
        return memory_data

    def save(self, memory_data: dict[str, Any], agent_name: str | None = None) -> bool:
        """Save memory data to file and update cache."""
        file_path = self._get_memory_file_path(agent_name)
        return self._save_to_file(memory_data, file_path, agent_name)

    # -- Per-user memory operations -----------------------------------------

    def load_user_memory(self, channel_name: str, user_id: str) -> dict[str, Any]:
        """Load per-user memory (cached)."""
        file_path = self._get_user_memory_file_path(channel_name, user_id)
        cache_key = (channel_name, user_id)
        return self._load_cached(cache_key, file_path)

    def reload_user_memory(self, channel_name: str, user_id: str) -> dict[str, Any]:
        """Force-reload per-user memory."""
        file_path = self._get_user_memory_file_path(channel_name, user_id)
        cache_key = (channel_name, user_id)
        memory_data = self._load_memory_from_file(file_path)

        try:
            mtime = file_path.stat().st_mtime if file_path.exists() else None
        except OSError:
            mtime = None

        self._memory_cache[cache_key] = (memory_data, mtime)
        return memory_data

    def save_user_memory(self, memory_data: dict[str, Any], channel_name: str, user_id: str) -> bool:
        """Save per-user memory."""
        file_path = self._get_user_memory_file_path(channel_name, user_id)
        cache_key = (channel_name, user_id)
        return self._save_to_file(memory_data, file_path, cache_key)

    def clear_user_memory(self, channel_name: str, user_id: str) -> bool:
        """Clear a specific user's memory."""
        return self.save_user_memory(create_empty_memory(), channel_name, user_id)

    def list_user_memory_files(self, channel_name: str) -> list[Path]:
        """List all per-user memory JSON files for a channel."""
        channel_dir = get_paths().channel_memory_dir(channel_name)
        if not channel_dir.exists():
            return []
        return sorted(channel_dir.glob("*.json"))


_storage_instance: MemoryStorage | None = None
_storage_lock = threading.Lock()


def get_memory_storage() -> MemoryStorage:
    """Get the configured memory storage instance."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    with _storage_lock:
        if _storage_instance is not None:
            return _storage_instance

        config = get_memory_config()
        storage_class_path = config.storage_class

        try:
            module_path, class_name = storage_class_path.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            storage_class = getattr(module, class_name)

            # Validate that the configured storage is a MemoryStorage implementation
            if not isinstance(storage_class, type):
                raise TypeError(f"Configured memory storage '{storage_class_path}' is not a class: {storage_class!r}")
            if not issubclass(storage_class, MemoryStorage):
                raise TypeError(f"Configured memory storage '{storage_class_path}' is not a subclass of MemoryStorage")

            _storage_instance = storage_class()
        except Exception as e:
            logger.error(
                "Failed to load memory storage %s, falling back to FileMemoryStorage: %s",
                storage_class_path,
                e,
            )
            _storage_instance = FileMemoryStorage()

    return _storage_instance
