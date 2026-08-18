"""
core/monitor.py — Concurrent multi-path monitoring orchestration.

This module is the Step 7 "monitoring layer". Its job is to take a list of
configured Path objects and collect health metrics for all of them concurrently.

Architecture overview
─────────────────────
The dependency direction is strictly one-way:

    monitor.py (orchestration)
        ↓  imports from
    loss.py, stats.py, throughput.py, path.py (primitives)

The primitives do NOT know this module exists.

Why asyncio + asyncio.to_thread()?
───────────────────────────────────
The existing networking primitives (run_probes, measure_throughput) are
synchronous: they use blocking socket calls and cannot be `await`-ed directly
in an asyncio event loop. Calling them directly inside a coroutine would block
the entire event loop for the duration of the network operation, serialising
what should be concurrent monitoring.

asyncio.to_thread(fn, *args) solves this cleanly:
  1. It submits `fn(*args)` to Python's default thread pool executor.
  2. The event loop is NOT blocked — it can run other coroutines while the
     blocking function waits on its sockets in a separate OS thread.
  3. When the thread finishes, the event loop resumes the awaiting coroutine.

This means we keep the synchronous probing primitives exactly as they are
(tested, documented, correct) and compose them into concurrent monitoring
without rewriting them.

Concurrency model
─────────────────
Two levels of concurrency are used:

  Level 1 — across paths:
    asyncio.gather(*[_monitor_one_path(p, ...) for p in paths])
    All paths are dispatched concurrently. One slow or failed path does not
    delay the others.

  Level 2 — within one path:
    For each path, four measurements are also dispatched concurrently:
      - run_probes()        → ProbeStats  (RTT samples + loss rate)
      - compute_rtt_stats() → RttStats    (jitter, mean, min, max) — pure, fast
      - measure_throughput()→ ThroughputResult
    Note: rtt_stats depends on the output of run_probes, so those two are
    sequential within a path. The throughput measurement runs concurrently
    alongside run_probes.

Trade-offs
──────────
  + No changes to existing synchronous primitives
  + Clean separation: orchestration here, I/O in the primitives
  + asyncio.gather() provides isolation: one path raising an exception is
    caught per-path, not allowed to cancel others (using return_exceptions=True
    internally via individual try/except in _monitor_one_path)
  - Each asyncio.to_thread() call spawns an OS thread. For N paths with
    concurrent probes+throughput per path, peak thread count ≈ 2N.
    For a prototype with 2–10 paths this is negligible.
  - The asyncio event loop must be running (callers use asyncio.run() or
    await from an existing async context). This is a deliberate design
    boundary: synchronous callers use asyncio.run(monitor_paths(...)).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.loss import ProbeStats, run_probes
from core.path import Path
from core.probe import ProbeStatus
from core.stats import RttStats, compute_rtt_stats
from core.throughput import ThroughputResult, measure_throughput

# ── Default monitoring parameters ─────────────────────────────────────────────
#
# These defaults are intentionally conservative:
#   - Small probe count: keeps test suites fast; real deployments will increase
#   - Small transfer: fast throughput measurement; real deployments use 1+ MiB
#
# All parameters are overridable on every call — no hidden global state.

DEFAULT_PROBE_COUNT: int = 3
DEFAULT_PROBE_TIMEOUT: float = 2.0
DEFAULT_TRANSFER_BYTES: int = 64 * 1024    # 64 KiB — small but exercices the path
DEFAULT_THROUGHPUT_TIMEOUT: float = 10.0


# ── Result types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PathMetrics:
    """
    All health metrics collected for a single monitored path.

    This is the per-path result. Every field is present regardless of whether
    measurements succeeded or failed — failure is represented in the status
    fields of the individual result objects, not by raising exceptions.

    Attributes
    ----------
    path:
        The Path configuration that was monitored. Preserved so callers can
        always look up which configuration produced these metrics, without
        needing to correlate by name string separately.
    probe_stats:
        Result of run_probes(): counts of success/failure and loss rate.
        None if the measurement could not be attempted (e.g., invalid config).
    rtt_stats:
        RTT statistics (mean, min, max, jitter) computed from successful probes.
        None if no probes succeeded (probe_stats.successful == 0) or if
        probe_stats itself is None.
    throughput:
        Result of measure_throughput(). Status field indicates success/failure.
        None if the measurement could not be attempted.
    error:
        If an unexpected exception escaped from this path's monitoring task,
        its string representation is stored here. This should not happen during
        normal operation — all expected failures are captured in the result
        objects above. Its presence indicates a programming error.

    Why frozen?
        Metrics are a snapshot of what was measured at a specific time.
        Mutating them after the fact would make results untrustworthy.
    """

    path: Path
    probe_stats: ProbeStats | None
    rtt_stats: RttStats | None
    throughput: ThroughputResult | None
    error: str | None = None

    def __str__(self) -> str:
        lines = [f"PathMetrics({self.path})"]
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        if self.probe_stats:
            lines.append(f"  loss={self.probe_stats.loss_rate:.1f}%")
        if self.rtt_stats and self.rtt_stats.count > 0:
            lines.append(
                f"  rtt=mean:{self.rtt_stats.mean_ms:.2f}ms "
                f"jitter:{self.rtt_stats.jitter_ms:.2f}ms"
            )
        if self.throughput and self.throughput.throughput_mbps is not None:
            lines.append(f"  throughput={self.throughput.throughput_mbps:.1f} Mbps")
        return "\n".join(lines)


@dataclass(frozen=True)
class MonitorResult:
    """
    The result of monitoring a set of paths.

    Attributes
    ----------
    metrics:
        A dict mapping path.name → PathMetrics for every path that was
        monitored. Path names are unique within a monitoring run.
        The dict is the canonical result; access via result.metrics["primary"].

    Why a dict keyed by name?
        Callers frequently need to look up "what were Path X's metrics?"
        A list would require iterating. A dict gives O(1) access and makes
        the association between path identity and metrics explicit.

    Why frozen?
        MonitorResult is a complete, immutable snapshot. Like a photo — you
        don't edit the photo after you take it.
    """

    metrics: dict[str, PathMetrics]

    def __str__(self) -> str:
        if not self.metrics:
            return "MonitorResult(no paths monitored)"
        lines = [f"MonitorResult({len(self.metrics)} paths):"]
        for name, m in self.metrics.items():
            lines.append(f"  [{name}] {m.path.host}:{m.path.port}")
            if m.probe_stats:
                lines.append(
                    f"    probe_loss={m.probe_stats.loss_rate:.1f}%  "
                    f"successful={m.probe_stats.successful}/{m.probe_stats.total}"
                )
            if m.rtt_stats and m.rtt_stats.count > 0:
                lines.append(
                    f"    rtt_mean={m.rtt_stats.mean_ms:.2f}ms  "
                    f"jitter={m.rtt_stats.jitter_ms:.2f}ms"
                )
            if m.throughput and m.throughput.throughput_mbps is not None:
                lines.append(
                    f"    throughput={m.throughput.throughput_mbps:.1f} Mbps"
                )
        return "\n".join(lines)


# ── Internal: single-path monitoring coroutine ────────────────────────────────


async def _monitor_one_path(
    path: Path,
    probe_count: int,
    probe_timeout: float,
    transfer_bytes: int,
    throughput_timeout: float,
) -> PathMetrics:
    """
    Collect all health metrics for one path, concurrently where possible.

    This is an internal async helper. It is called by monitor_paths() via
    asyncio.gather(), which runs one instance per path concurrently.

    Measurement scheduling
    ──────────────────────
    Within a single path, we schedule two blocking operations concurrently:

      Task A: run_probes()          — blocking, runs in a thread
      Task B: measure_throughput()  — blocking, runs in a thread

    Both are dispatched simultaneously with asyncio.gather(). The event loop
    can context-switch between them (and between this path and other paths)
    while they block on sockets.

    After both complete:
      compute_rtt_stats(probe_stats.rtt_values_ms) is called directly (it is
      a pure math function, not I/O, and runs in microseconds — no thread needed).

    Why not also run rtt_stats concurrently?
      compute_rtt_stats() takes the output of run_probes() as input. It cannot
      start until run_probes() finishes. Sequential dependency, not a choice.
      It is also a pure function with no I/O, so it should NOT be wrapped in
      asyncio.to_thread() — that would add unnecessary thread overhead.

    Failure isolation
    ─────────────────
    Any exception that escapes run_probes() or measure_throughput() is caught
    here. The exception is recorded in PathMetrics.error, and None is stored
    for the affected measurement field. The caller (monitor_paths) always
    receives a PathMetrics, never an exception from this path.

    Parameters
    ----------
    path:
        The path to monitor.
    probe_count:
        Number of RTT probes to run via run_probes().
    probe_timeout:
        Per-probe timeout forwarded to run_probes().
    transfer_bytes:
        Bytes to transfer in the throughput measurement.
    throughput_timeout:
        Socket timeout forwarded to measure_throughput().

    Returns
    -------
    PathMetrics
        Always returns — never raises.
    """
    probe_stats: ProbeStats | None = None
    rtt_stats: RttStats | None = None
    throughput: ThroughputResult | None = None
    error: str | None = None

    try:
        # Dispatch both blocking measurements concurrently in the thread pool.
        # asyncio.gather() suspends this coroutine until both threads finish.
        probe_result, throughput_result = await asyncio.gather(
            asyncio.to_thread(
                run_probes,
                path.host,
                path.port,
                probe_count,
                probe_timeout,
            ),
            asyncio.to_thread(
                measure_throughput,
                path.host,
                path.port,
                transfer_bytes,
                64 * 1024,      # chunk_size: 64 KiB
                throughput_timeout,
            ),
        )

        probe_stats = probe_result
        throughput = throughput_result

        # RTT statistics are computed from the probe results.
        # This is a pure math function — no I/O, no thread needed.
        rtt_stats = compute_rtt_stats(probe_stats.rtt_values_ms)

    except Exception as exc:  # noqa: BLE001 — intentional broad catch for isolation
        # This should not happen during normal operation.
        # run_probes() and measure_throughput() are designed to never raise
        # on network errors — they return error results instead.
        # If we reach here, something unexpected happened (programming error,
        # resource exhaustion, etc.). We record it and continue.
        error = f"{type(exc).__name__}: {exc}"

    return PathMetrics(
        path=path,
        probe_stats=probe_stats,
        rtt_stats=rtt_stats,
        throughput=throughput,
        error=error,
    )


# ── Public API ────────────────────────────────────────────────────────────────


async def monitor_paths(
    paths: list[Path],
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    transfer_bytes: int = DEFAULT_TRANSFER_BYTES,
    throughput_timeout: float = DEFAULT_THROUGHPUT_TIMEOUT,
) -> MonitorResult:
    """
    Monitor all configured paths concurrently and return a MonitorResult.

    All paths are dispatched simultaneously. The total wall-clock time for
    monitoring N paths is approximately equal to the slowest single path,
    not the sum of all paths.

    Parameters
    ----------
    paths:
        List of Path objects to monitor. An empty list is valid — the result
        will have an empty metrics dict. Order within the list does not matter.
    probe_count:
        Number of RTT probes to run per path. Default: 3.
        Increase for more accurate statistics; decrease for faster tests.
    probe_timeout:
        Per-probe socket timeout in seconds. Default: 2.0s.
        Probes that exceed this timeout contribute to probe loss rate.
    transfer_bytes:
        Bytes of data to transfer in the throughput measurement. Default: 64 KiB.
        Increase for more stable throughput readings; decrease for faster tests.
    throughput_timeout:
        Socket timeout for the throughput transfer in seconds. Default: 10.0s.

    Returns
    -------
    MonitorResult
        A snapshot of metrics for all paths. The .metrics dict maps each
        path's name to its PathMetrics. Always returns — never raises.

    Example (synchronous caller)
    ────────────────────────────
        import asyncio
        from core.path import Path
        from core.monitor import monitor_paths

        paths = [
            Path(name="primary", host="10.0.0.1", port=5201),
            Path(name="backup",  host="10.0.0.2", port=5201),
        ]
        result = asyncio.run(monitor_paths(paths))
        print(result.metrics["primary"].probe_stats.loss_rate)

    Notes
    ─────
    Path names must be unique within the paths list. If duplicate names are
    present, the last one wins in the result dict (standard Python dict
    semantics). Validation of uniqueness is the caller's responsibility.
    """
    if not paths:
        # Empty input → empty result. Valid, not an error.
        return MonitorResult(metrics={})

    # Dispatch one monitoring coroutine per path.
    # asyncio.gather() runs all coroutines concurrently:
    #   - Each coroutine calls asyncio.to_thread() internally, handing the
    #     blocking socket work to OS threads.
    #   - The event loop services all paths simultaneously.
    #   - return_exceptions=False is the default; individual path failures are
    #     caught inside _monitor_one_path, so no exception should escape here.
    path_metrics_list: list[PathMetrics] = await asyncio.gather(
        *[
            _monitor_one_path(
                path=p,
                probe_count=probe_count,
                probe_timeout=probe_timeout,
                transfer_bytes=transfer_bytes,
                throughput_timeout=throughput_timeout,
            )
            for p in paths
        ]
    )

    # Build the name → PathMetrics dict.
    # We use path.name as the key so callers can do result.metrics["primary"]
    # instead of searching through a list.
    metrics: dict[str, PathMetrics] = {
        pm.path.name: pm for pm in path_metrics_list
    }

    return MonitorResult(metrics=metrics)
