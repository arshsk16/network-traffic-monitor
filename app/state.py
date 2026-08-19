"""
app/state.py — Application-level shared state for the monitoring API.

Why this exists
───────────────
FastAPI creates a new router function call per HTTP request, but the
PathState object must persist across requests for correct failover
detection. If we created a fresh PathState on every request:

    Request 1: None → primary  →  INITIAL_SELECTION   (correct)
    Request 2: None → primary  →  INITIAL_SELECTION   (WRONG — no memory)
    Request 3: None → backup   →  INITIAL_SELECTION   (WRONG — not FAILOVER)

By keeping one shared PathState and one shared list of demo paths in this
module, all requests share the same monitoring state:

    Request 1: None → primary  →  INITIAL_SELECTION
    Request 2: primary → primary → NO_CHANGE
    Request 3: primary → backup  → FAILOVER

How state is kept safe
──────────────────────
FastAPI is async. run_monitoring_cycle() is async. Python's asyncio event
loop is single-threaded, so awaiting run_monitoring_cycle() in a route
function is safe without a lock: only one coroutine can be running at a
time. No mutex or threading.Lock is needed for the monitoring call itself.

For a production service that handles many concurrent poll requests, a
proper lock or request queue would be added. For this prototype, the
async model is sufficient.

Demo paths
──────────
We start two _DemoServer instances at import time. They run as daemon
threads and are cleaned up automatically when the process exits.

The same _DemoServer class from app.cli is reused — no duplication.
"""

from __future__ import annotations

from app.cli import _DemoServer
from core.failover import PathState
from core.path import Path

# ── Start demo servers ────────────────────────────────────────────────────────
# Two loopback servers started once at module import (process startup).
# They persist for the lifetime of the FastAPI process.

_primary_server = _DemoServer("primary")
_backup_server = _DemoServer("backup")
_primary_server.start()
_backup_server.start()

# ── Shared monitoring state ───────────────────────────────────────────────────
# One PathState instance, shared by all API requests.
# Imported by the monitor router via:  from app.state import monitoring_state, demo_paths

monitoring_state: PathState = PathState()

demo_paths: list[Path] = [_primary_server.path, _backup_server.path]

# Expose the server objects so they can be stopped in tests if needed.
primary_server: _DemoServer = _primary_server
backup_server: _DemoServer = _backup_server
