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

Demo controls
─────────────
simulate_primary_failure()
    Stops the primary server so monitoring probes see connection-refused
    → 100% loss → scoring marks primary unavailable → FAILOVER.
    Sets primary_failed = True.

reset_demo()
    Restarts the primary server on a new ephemeral port, rebuilds
    demo_paths in-place, and resets the PathState so the sequence
    INITIAL_SELECTION → NO_CHANGE → FAILOVER → NO_CHANGE can be
    demonstrated from the beginning again.
    Sets primary_failed = False.
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
# The monitor router accesses this via:  import app.state as app_state
#   then  app_state.monitoring_state  (module attribute lookup each call)
# so that reset_demo() reassignments are immediately visible to endpoints.

monitoring_state: PathState = PathState()

# demo_paths is mutated in-place by reset_demo() so any reference to this
# list (e.g., in tests) always sees the current server addresses.
demo_paths: list[Path] = [_primary_server.path, _backup_server.path]

# ── Demo control flag ─────────────────────────────────────────────────────────
# True while the primary server has been deliberately stopped for the demo.
# Used by the API to reflect demo state to the UI.
primary_failed: bool = False

# Expose the server objects so they can be stopped in tests if needed.
primary_server: _DemoServer = _primary_server
backup_server: _DemoServer = _backup_server


# ── Demo control helpers ──────────────────────────────────────────────────────


def simulate_primary_failure() -> None:
    """
    Stop the primary demo server so that the next monitoring cycle sees
    connection-refused → 100% probe loss → path marked unavailable →
    FAILOVER transition to backup.

    This exercises the real monitoring pipeline — no metrics are faked.
    """
    global primary_failed
    _primary_server.stop()
    primary_failed = True


def reset_demo() -> None:
    """
    Restore the demo environment so the four-cycle demonstration
    (INITIAL_SELECTION → NO_CHANGE → FAILOVER → NO_CHANGE) can be
    repeated from the beginning.

    Steps:
      1. Stop primary server (idempotent if already stopped).
      2. Restart primary server on a new ephemeral loopback port.
      3. Rebuild demo_paths in-place with the new server address.
      4. Replace monitoring_state with a fresh PathState() at module level.
         The monitor router reads app_state.monitoring_state on every call
         (module-attribute lookup) so the new object is used immediately.
      5. Clear the primary_failed flag.
    """
    global monitoring_state, primary_failed

    # Restart primary server (stop is idempotent)
    _primary_server.stop()
    _primary_server.start()

    # Rebuild the path list in-place so all references see the new address
    demo_paths[:] = [_primary_server.path, _backup_server.path]

    # Replace the PathState binding at module level.
    monitoring_state = PathState()

    primary_failed = False
