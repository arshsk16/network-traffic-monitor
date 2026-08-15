"""
Entry point for running the development server directly.

Usage:
    python run.py

This is a convenience script so you don't have to remember the full
uvicorn command during development. Do NOT use this in production — use
a proper process manager (e.g. gunicorn + uvicorn workers) instead.
"""

import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
