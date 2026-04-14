"""Configuration for memory mechanism."""

from pydantic import BaseModel, Field


class PerUserMemoryConfig(BaseModel):
    """Per-platform toggle for per-user memory isolation.

    When a platform is enabled here, each user on that platform gets their own
    memory file (e.g. ``memory/feishu/{open_id}.json``) and the global memory
    file is **not** used for that platform.  Platforms that are disabled (or not
    listed) fall back to the shared ``memory/global.json``.
    """

    feishu: bool = Field(
        default=False,
        description="Enable per-user memory for Feishu channel (uses sender open_id)",
    )


class MemoryCleanupConfig(BaseModel):
    """Configuration for scheduled memory cleanup."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable periodic memory cleanup",
    )
    interval_hours: int = Field(
        default=168,
        ge=1,
        description="Cleanup interval in hours (default: 168 = 7 days)",
    )
    apply_to_global: bool = Field(
        default=False,
        description="Whether cleanup also applies to the global memory file",
    )


class MemoryConfig(BaseModel):
    """Configuration for global memory mechanism."""

    enabled: bool = Field(
        default=True,
        description="Whether to enable memory mechanism",
    )
    storage_path: str = Field(
        default="",
        description=(
            "Path to store memory data. "
            "If empty, defaults to `{base_dir}/memory/global.json` (see Paths.global_memory_file). "
            "Absolute paths are used as-is. "
            "Relative paths are resolved against `Paths.base_dir` "
            "(not the backend working directory). "
            "Note: if you previously set this to `.deer-flow/memory.json`, "
            "the file will now be resolved as `{base_dir}/.deer-flow/memory.json`; "
            "migrate existing data or use an absolute path to preserve the old location."
        ),
    )
    storage_class: str = Field(
        default="deerflow.agents.memory.storage.FileMemoryStorage",
        description="The class path for memory storage provider",
    )
    debounce_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Seconds to wait before processing queued updates (debounce)",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name to use for memory updates (None = use default model)",
    )
    max_facts: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of facts to store",
    )
    fact_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for storing facts",
    )
    injection_enabled: bool = Field(
        default=True,
        description="Whether to inject memory into system prompt",
    )
    max_injection_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Maximum tokens to use for memory injection",
    )
    per_user: PerUserMemoryConfig = Field(
        default_factory=PerUserMemoryConfig,
        description="Per-platform per-user memory isolation settings",
    )
    cleanup: MemoryCleanupConfig = Field(
        default_factory=MemoryCleanupConfig,
        description="Scheduled memory cleanup settings",
    )


# Global configuration instance
_memory_config: MemoryConfig = MemoryConfig()


def get_memory_config() -> MemoryConfig:
    """Get the current memory configuration."""
    return _memory_config


def set_memory_config(config: MemoryConfig) -> None:
    """Set the memory configuration."""
    global _memory_config
    _memory_config = config


def is_per_user_memory_enabled(channel_name: str | None) -> bool:
    """Check whether per-user memory isolation is enabled for *channel_name*.

    Returns ``False`` when ``channel_name`` is ``None`` (e.g. web UI) or when
    the channel is not configured for per-user memory.
    """
    if not channel_name:
        return False
    config = get_memory_config()
    return getattr(config.per_user, channel_name, False)


def load_memory_config_from_dict(config_dict: dict) -> None:
    """Load memory configuration from a dictionary."""
    global _memory_config
    _memory_config = MemoryConfig(**config_dict)
