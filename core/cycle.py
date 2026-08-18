"""
core/cycle.py — End-to-end monitoring cycle orchestration.

This module is the Step 10 "integration layer". It wires together the four
existing components into one complete monitoring cycle:

    paths → monitor → score → state update → CycleResult

Architecture overview
─────────────────────
Dependency direction is strictly one-way:

    cycle.py (orchestration)
        ↓  calls
    monitor.py     — concurrent path monitoring  (Step 7)
    scoring.py     — weighted path scoring       (Step 8)
    failover.py    — preferred-path state mgmt   (Step 9)

The primitive layers (probe.py, loss.py, stats.py, throughput.py, path.py)
are not imported here — they are already encapsulated inside monitor.py.

What this module does
─────────────────────
Exposes one public async function:

    run_monitoring_cycle(paths, state, ...) → CycleResult

A CycleResult bundles the complete output of one cycle:
    - MonitorResult   (raw measurements, from Step 7)
    - ScoringResult   (normalized scores + preferred path, from Step 8)
    - PathTransition  (state change classification, from Step 9)

What this module does NOT do
────────────────────────────
- Does NOT re-implement TCP probing, RTT, loss, jitter, or throughput
- Does NOT re-implement normalization or weighted scoring
- Does NOT re-implement state management or transition classification
- Does NOT modify OS routing tables
- Does NOT open any sockets directly
- Does NOT expose a REST endpoint (that is a later step)
- Does NOT run a background service or scheduler
- Does NOT implement retries, hysteresis, or cooldowns

Every piece of logic lives in one of the four existing modules.
This module is pure sequencing and packaging.

Why async?
──────────
monitor_paths() (Step 7) is an async coroutine because it uses asyncio.gather()
to run blocking socket operations concurrently via asyncio.to_thread(). The
integration layer must await it, so run_monitoring_cycle() is also async.

scoring and state-update steps are synchronous pure functions — they run in the
event loop directly (no I/O, microsecond duration).

Callers from synchronous contexts use:
    import asyncio
    result = asyncio.run(run_monitoring_cycle(paths, state))

Callers from async contexts (FastAPI, other async code):
    result = await run_monitoring_cycle(paths, state)

Error handling philosophy
─────────────────────────
Network failures are already handled inside monitor.py and represented as
non-SUCCESS status in result objects — they do not raise exceptions. The
integration layer does not need to catch them.

Programming errors (bugs, misconfiguration, import errors) should remain
visible — the integration layer does NOT broadly catch Exception and swallow
them. If run_monitoring_cycle() raises, it is either:
  1. A programming error that needs fixing
  2. A configuration error (invalid Path, invalid ScoringWeights)
Both should propagate to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.failover import PathState, PathTransition
from core.monitor import (
    DEFAULT_PROBE_COUNT,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_THROUGHPUT_TIMEOUT,
    DEFAULT_TRANSFER_BYTES,
    MonitorResult,
    monitor_paths,
)
from core.path import Path
from core.scoring import ScoringResult, ScoringWeights, score_paths


# ── Cycle result ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CycleResult:
    """
    The complete, immutable result of one monitoring cycle.

    Bundles the output of all three pipeline stages so callers have a single
    object from which they can answer any question about what happened:

    "What paths were monitored?"
        → cycle.monitoring_result.metrics.keys()

    "What measurements did path X produce?"
        → cycle.monitoring_result.metrics["X"].probe_stats
        → cycle.monitoring_result.metrics["X"].rtt_stats
        → cycle.monitoring_result.metrics["X"].throughput

    "How was path X scored?"
        → next(sp for sp in cycle.scoring_result.scored_paths
               if sp.path.name == "X")
        → .rtt_score, .loss_score, .jitter_score, .throughput_score, .total_score

    "Which path is currently preferred?"
        → cycle.scoring_result.preferred_path
        → cycle.transition.new_path

    "Was there a failover?"
        → cycle.transition.event_type == TransitionType.FAILOVER
        → cycle.transition.is_failover()

    "What happened to the previous preferred path?"
        → cycle.transition.previous_path
        → cycle.transition.previous_scored_path

    Attributes
    ----------
    monitoring_result:
        The raw per-path measurements produced by monitor_paths().
        Preserved in full — RTT values, loss rate, throughput, etc.
        Included even for failed paths (they show 100% loss, etc.).
    scoring_result:
        The normalized scores and ranking produced by score_paths().
        Includes all paths (available and unavailable).
        preferred_path is the current best available path, or None.
    transition:
        The state-change record produced by state.update().
        event_type classifies what changed:
          INITIAL_SELECTION  — first path ever chosen
          NO_CHANGE          — same path as before
          FAILOVER           — preferred path changed
          NO_AVAILABLE_PATH  — no usable path exists

    Why frozen?
        A CycleResult is a snapshot of one point in time. Making it mutable
        would allow silent corruption of historical cycle records.

    Why not expose just preferred_path as a string?
        Reducing the result to a single string discards all the information
        needed to understand WHY the decision was made. The full result lets
        callers log detailed diagnostics, drive REST API responses, feed
        dashboards, and write targeted tests without re-running measurements.
    """

    monitoring_result: MonitorResult
    scoring_result: ScoringResult
    transition: PathTransition

    # ── Convenience accessors ──────────────────────────────────────────────────

    @property
    def preferred_path_name(self) -> str | None:
        """
        Shortcut for the currently preferred path's name, or None.

        Equivalent to:
            cycle.transition.new_path.name if cycle.transition.new_path else None
        """
        if self.transition.new_path is None:
            return None
        return self.transition.new_path.name

    @property
    def is_failover(self) -> bool:
        """Shortcut: True if this cycle produced a FAILOVER transition."""
        return self.transition.is_failover()

    @property
    def is_change(self) -> bool:
        """Shortcut: True if the preferred path changed in any way this cycle."""
        return self.transition.is_change()

    def __str__(self) -> str:
        n_paths = len(self.monitoring_result.metrics)
        preferred = self.preferred_path_name or "none"
        event = self.transition.event_type.value
        return (
            f"CycleResult("
            f"paths={n_paths}, "
            f"preferred={preferred!r}, "
            f"event={event})"
        )


# ── Public API ─────────────────────────────────────────────────────────────────


async def run_monitoring_cycle(
    paths: list[Path],
    state: PathState,
    *,
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    transfer_bytes: int = DEFAULT_TRANSFER_BYTES,
    throughput_timeout: float = DEFAULT_THROUGHPUT_TIMEOUT,
    weights: ScoringWeights | None = None,
) -> CycleResult:
    """
    Execute one complete monitoring cycle and return a CycleResult.

    This function is the entry point for the integrated monitoring pipeline.
    It performs three sequential operations:

      Step 1 — Monitor (async, concurrent):
          Calls monitor_paths() to measure all configured paths concurrently.
          Returns MonitorResult with RTT, loss, jitter, and throughput per path.

      Step 2 — Score (sync, pure):
          Calls score_paths() to normalize metrics and compute weighted scores.
          Returns ScoringResult with rankings and the current preferred path.

      Step 3 — State update (sync, stateful):
          Calls state.update() to classify the transition and update the
          persistent preferred-path record.
          Returns PathTransition describing what changed.

    The same PathState instance must be passed on every successive call to
    preserve continuity across cycles. The state is mutated in-place by
    state.update() — this is intentional and documented in PathState.

    Parameters
    ----------
    paths:
        The list of logical paths to monitor. An empty list is valid —
        it will produce a cycle with no measurements and NO_AVAILABLE_PATH.
    state:
        The PathState instance to update. Must be the SAME instance across
        cycles for correct failover detection. Create once; reuse always.
    probe_count:
        Number of TCP RTT probes per path. Default: 3.
        Increase for more accurate statistics.
    probe_timeout:
        Per-probe TCP connect timeout in seconds. Default: 2.0.
    transfer_bytes:
        Bytes to transfer in the throughput measurement. Default: 64 KiB.
    throughput_timeout:
        Timeout for the throughput measurement in seconds. Default: 10.0.
    weights:
        ScoringWeights for the weighted path score. If None, uses the
        default weights (RTT=0.30, loss=0.30, jitter=0.20, throughput=0.20).
        Pass a custom ScoringWeights instance to adjust priorities.

    Returns
    -------
    CycleResult
        Contains monitoring_result, scoring_result, and transition.
        Always returns — does not raise on network failures (those are
        represented inside MonitorResult as non-SUCCESS statuses).

    Raises
    ------
    ValueError
        If weights is invalid (negative values, wrong sum). Raised by
        ScoringWeights.__post_init__ before any monitoring occurs.
    TypeError / other
        If paths contains non-Path objects or state is not a PathState.
        Programming errors propagate — they are not swallowed.

    Example — synchronous caller
    ────────────────────────────
        import asyncio
        from core.path import Path
        from core.failover import PathState
        from core.cycle import run_monitoring_cycle

        paths = [
            Path(name="primary", host="10.0.0.1", port=9001),
            Path(name="backup",  host="10.0.0.2", port=9001),
        ]
        state = PathState()

        # Cycle 1 — initial selection
        result = asyncio.run(run_monitoring_cycle(paths, state))
        print(result.transition.event_type)      # INITIAL_SELECTION
        print(result.preferred_path_name)        # "primary" (or "backup")

        # Cycle 2 — no change (if path quality is stable)
        result = asyncio.run(run_monitoring_cycle(paths, state))
        print(result.transition.event_type)      # NO_CHANGE

    Example — async caller (e.g., FastAPI background task)
    ───────────────────────────────────────────────────────
        state = PathState()
        while True:
            result = await run_monitoring_cycle(paths, state)
            if result.is_failover:
                logger.warning("Failover: %s → %s",
                               result.transition.previous_path,
                               result.transition.new_path)
            await asyncio.sleep(30)

    Notes
    ─────
    - The state parameter is mutated by state.update(). This is intentional:
      the state must persist across calls to track transitions correctly.
    - scoring and state.update() are synchronous. They run in the event loop
      thread (not in asyncio.to_thread) because they perform no I/O.
    - monitor_paths() is the only async operation in the cycle; it dispatches
      blocking socket operations to the thread pool via asyncio.to_thread().
    """
    # ── Step 1: Monitor all paths concurrently ─────────────────────────────────
    # monitor_paths() handles all networking; failures are captured in result
    # objects, not raised as exceptions. Safe to await directly.
    monitoring_result: MonitorResult = await monitor_paths(
        paths,
        probe_count=probe_count,
        probe_timeout=probe_timeout,
        transfer_bytes=transfer_bytes,
        throughput_timeout=throughput_timeout,
    )

    # ── Step 2: Score the monitoring results ───────────────────────────────────
    # score_paths() is a pure synchronous function. No I/O. Runs in the event
    # loop thread (no thread overhead). Produces normalized scores and ranking.
    scoring_result: ScoringResult = score_paths(monitoring_result, weights)

    # ── Step 3: Update failover state ─────────────────────────────────────────
    # state.update() is synchronous and stateful. It compares the new preferred
    # path to the previously remembered one and returns an immutable transition.
    transition: PathTransition = state.update(scoring_result)

    # ── Step 4: Return bundled result ─────────────────────────────────────────
    return CycleResult(
        monitoring_result=monitoring_result,
        scoring_result=scoring_result,
        transition=transition,
    )
