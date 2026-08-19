"""
app/routers/monitor.py — FastAPI router for the monitoring API.

Exposes:
    GET  /api/v1/monitor               — run one monitoring cycle, return JSON
    POST /api/v1/demo/simulate-failure — stop primary server (real failure)
    POST /api/v1/demo/reset            — restart primary server + reset state

Architecture
────────────
    HTTP GET /api/v1/monitor
           ↓
    run_monitoring_cycle(demo_paths, monitoring_state)   ← Step 10
           ↓
    CycleResult
           ↓
    MonitorResponse (Pydantic)
           ↓
    JSON

This router contains ZERO:
  - socket logic
  - scoring formulas
  - failover decisions
  - monitoring business logic

It calls run_monitoring_cycle() and converts CycleResult → Pydantic model.

State management
────────────────
monitoring_state (PathState) and demo_paths (list[Path]) are accessed via
`import app.state as app_state` and read as module-level attributes on each
call. This ensures that reset_demo() reassignments (which rebind the module-
level `monitoring_state` name) are immediately visible to all endpoints
without requiring a process restart.

Error handling
──────────────
Expected network failures are represented inside CycleResult as
unavailable paths. They do not raise exceptions and therefore do not
cause 500 errors. An unexpected programming error (e.g. config bug)
will propagate as a 500 naturally — we do NOT catch Exception broadly.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

import app.state as app_state
from core.cycle import CycleResult, run_monitoring_cycle

router = APIRouter(prefix="/api/v1", tags=["monitor"])


# ── Response models ────────────────────────────────────────────────────────────


class PathMetricsResponse(BaseModel):
    """Per-path metrics in the API response."""

    available: bool = Field(description="True if the path is reachable and measurable")
    rtt_ms: float | None = Field(
        description="Mean round-trip time in milliseconds (null if unavailable)"
    )
    loss_percent: float | None = Field(
        description="Probe loss rate as a percentage 0–100 (null if unavailable)"
    )
    jitter_ms: float | None = Field(
        description="RTT jitter in milliseconds (null if unavailable)"
    )
    throughput_mbps: float | None = Field(
        description=(
            "Application-level TCP throughput in Mbps measured in local test "
            "environment. This is NOT WAN bandwidth."
        )
    )
    score: float = Field(
        description="Weighted composite score 0–100 (0 = worst, 100 = best)"
    )
    rank: int = Field(description="Path rank among available paths (1 = preferred)")


class MonitorResponse(BaseModel):
    """
    Complete response for one monitoring cycle.

    All metric values are produced by the actual monitoring pipeline.
    None of them are hard-coded or estimated.
    """

    paths: dict[str, PathMetricsResponse] = Field(
        description="Per-path measurements and scores, keyed by path name"
    )
    preferred_path: str | None = Field(
        description="Name of the currently preferred path, or null if none available"
    )
    event: str = Field(
        description=(
            "State transition event: INITIAL_SELECTION | NO_CHANGE | "
            "FAILOVER | NO_AVAILABLE_PATH"
        )
    )
    previous_path: str | None = Field(
        description="Name of the previously preferred path (null on first cycle)"
    )
    cycle_note: str = Field(
        default=(
            "Local loopback demonstration. Throughput is application-level TCP "
            "throughput in the local test environment, not WAN bandwidth."
        ),
        description="Disclaimer note shown on every response",
    )
    primary_failed: bool = Field(
        default=False,
        description="True when the primary server has been deliberately stopped for the demo",
    )


class DemoControlResponse(BaseModel):
    """Response returned by demo control endpoints."""

    ok: bool = Field(description="True if the operation succeeded")
    message: str = Field(description="Human-readable description of what happened")
    primary_failed: bool = Field(
        description="Current value of the primary_failed flag after the operation"
    )


# ── Serialization helper ───────────────────────────────────────────────────────


def _cycle_result_to_response(result: CycleResult) -> MonitorResponse:
    """
    Convert a CycleResult into a MonitorResponse.

    This is the serialization boundary between the core data model and the
    API contract. It is the only place CycleResult attributes are read for
    the purpose of producing JSON.

    The conversion is mechanical:
      - scored_paths → per-path dict
      - preferred_path.path.name → preferred_path string
      - transition.event_type.value → event string
      - transition.previous_path.name → previous_path string
    """
    paths: dict[str, PathMetricsResponse] = {}

    for sp in result.scoring_result.scored_paths:
        name = sp.path.name
        pm = result.monitoring_result.metrics.get(name)

        # Extract raw values — None if the path is unavailable
        rtt_ms: float | None = None
        loss_percent: float | None = None
        jitter_ms: float | None = None
        throughput_mbps: float | None = None

        if pm is not None:
            if pm.probe_stats is not None:
                loss_percent = round(pm.probe_stats.loss_rate, 2)

            if pm.rtt_stats is not None and pm.rtt_stats.count > 0:
                rtt_ms = round(pm.rtt_stats.mean_ms, 3) if pm.rtt_stats.mean_ms is not None else None
                jitter_ms = round(pm.rtt_stats.jitter_ms, 3) if pm.rtt_stats.jitter_ms is not None else None

            if pm.throughput is not None and pm.throughput.throughput_mbps is not None:
                throughput_mbps = round(pm.throughput.throughput_mbps, 2)

        paths[name] = PathMetricsResponse(
            available=sp.available,
            rtt_ms=rtt_ms,
            loss_percent=loss_percent,
            jitter_ms=jitter_ms,
            throughput_mbps=throughput_mbps,
            score=round(sp.total_score, 2),
            rank=sp.rank,
        )

    preferred = (
        result.scoring_result.preferred_path.path.name
        if result.scoring_result.preferred_path is not None
        else None
    )
    previous = (
        result.transition.previous_path.name
        if result.transition.previous_path is not None
        else None
    )

    return MonitorResponse(
        paths=paths,
        preferred_path=preferred,
        event=result.transition.event_type.value,
        previous_path=previous,
        primary_failed=app_state.primary_failed,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.get(
    "/monitor",
    response_model=MonitorResponse,
    summary="Run one monitoring cycle",
    description=(
        "Execute a complete monitoring cycle: probe all configured paths, "
        "score them, update failover state, and return structured results. "
        "State persists across calls — repeated calls produce NO_CHANGE when "
        "the path quality is stable, and FAILOVER when the preferred path "
        "becomes unavailable."
    ),
)
async def run_monitor_cycle() -> MonitorResponse:
    """
    Run one complete monitoring cycle and return path metrics + state.

    Reads demo_paths and monitoring_state from the app.state module on
    each invocation so that reset_demo() changes take effect immediately.
    """
    result: CycleResult = await run_monitoring_cycle(
        app_state.demo_paths,
        state=app_state.monitoring_state,
        probe_count=2,
        probe_timeout=1.0,
        transfer_bytes=32 * 1024,
        throughput_timeout=5.0,
    )
    return _cycle_result_to_response(result)


@router.post(
    "/demo/simulate-failure",
    response_model=DemoControlResponse,
    summary="Simulate primary path failure",
    description=(
        "Stop the primary demo server so that the next monitoring cycle "
        "detects 100% probe loss on the primary path and triggers a FAILOVER "
        "transition to backup. No metrics are faked — the monitoring pipeline "
        "runs against a genuinely unavailable server."
    ),
)
async def demo_simulate_failure() -> DemoControlResponse:
    """Stop the primary demo server to trigger a real failover on next cycle."""
    app_state.simulate_primary_failure()
    return DemoControlResponse(
        ok=True,
        message="Primary server stopped. Run a monitoring cycle to observe FAILOVER.",
        primary_failed=app_state.primary_failed,
    )


@router.post(
    "/demo/reset",
    response_model=DemoControlResponse,
    summary="Reset the demo environment",
    description=(
        "Restart the primary demo server on a new ephemeral port and reset "
        "the monitoring PathState to None. The next cycle will produce "
        "INITIAL_SELECTION, allowing the full demonstration sequence "
        "(INITIAL_SELECTION → NO_CHANGE → FAILOVER → NO_CHANGE) to be "
        "repeated from scratch."
    ),
)
async def demo_reset() -> DemoControlResponse:
    """Restart the primary server and reset PathState for a clean demo run."""
    app_state.reset_demo()
    return DemoControlResponse(
        ok=True,
        message="Demo reset. Primary server restarted. Run a cycle to see INITIAL_SELECTION.",
        primary_failed=app_state.primary_failed,
    )

