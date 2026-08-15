"""
Application settings loaded from environment variables (or .env file).

Uses pydantic-settings so every setting is typed and validated at startup.
Add new settings here rather than scattering os.getenv() calls across the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the netmon service."""

    # ── Server ────────────────────────────────────────────────────────────────
    app_name: str = "netmon"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Future: Monitoring (placeholders — not yet implemented) ───────────────
    # probe_interval_seconds: float = 5.0
    # probe_timeout_seconds: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Single shared instance — import this everywhere instead of constructing Settings()
settings = Settings()
