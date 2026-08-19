"""
Application factory for the netmon FastAPI service.

Defines create_app() rather than a bare module-level `app` object so that:
  1. Tests can call create_app() to get a fresh instance without side effects.
  2. Settings overrides can be injected cleanly in tests.
  3. The startup/shutdown lifecycle is explicit and easy to extend later.
"""

from fastapi import FastAPI

from app.config import settings
from app.routers import dashboard, health, monitor


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Network Traffic Monitoring & Intelligent Path Selection — "
            "a learning project inspired by SD-WAN path-selection concepts."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(monitor.router)
    app.include_router(dashboard.router)

    return app


# Module-level app instance used by uvicorn:  uvicorn app.main:app
app = create_app()
