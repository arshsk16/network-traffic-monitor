"""
app/routers/dashboard.py — Serves the browser dashboard HTML.

GET /   →  app/static/index.html

A separate router for the dashboard keeps the concern isolated: if the
UI is later upgraded to a JS framework, only this file needs to change.
"""

from __future__ import annotations

from pathlib import Path as FilePath

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_HTML_PATH = FilePath(__file__).parent.parent / "static" / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the browser monitoring dashboard."""
    return HTMLResponse(content=_HTML_PATH.read_text(encoding="utf-8"))
