"""
tests/test_cli.py — Deterministic tests for app/cli.py.

Test strategy
─────────────
The CLI's output functions accept an `out: IO[str]` parameter so we can
pass a StringIO instead of sys.stdout. This lets us:
  - Capture all output as a string
  - Assert on specific substrings without any real terminal
  - Run entirely without a TTY (ANSI codes do not affect assertions because
    we search for plain text content, which is always present with or
    without ANSI wrappers)

For tests that need a real monitoring cycle (to verify data flow from
CycleResult into the formatted output), we start _DemoServer instances
locally and run run_monitoring_cycle() against them. These follow the
exact same pattern as test_cycle.py.

For tests that only check formatting logic, we construct minimal
CycleResult objects manually — no network operations needed.

What is NOT tested
──────────────────
- Exact throughput numbers (vary with machine speed)
- Exact RTT values (vary with scheduling)
- ANSI escape codes (they wrap plain text; plain content is asserted)
- The banner ASCII art character-for-character (visual only)

What IS tested
──────────────
- CLI imports and executes without error
- Header/banner contains expected title text
- Path names appear in per-path rows
- RTT values appear (as numbers with "ms" suffix)
- Loss values appear (as "%" values)
- Jitter values appear (as numbers with "ms" suffix)
- Throughput values appear (as "Mbps" values)
- Scores appear
- Preferred path is marked and named
- INITIAL_SELECTION label appears on first cycle
- NO_CHANGE label appears on repeated cycle
- FAILOVER label appears when path dies
- Transition "A → B" appears on failover
- Unavailable paths show "UNAVAILABLE"
- All four demo cycles complete without raising
"""

from __future__ import annotations

import asyncio
import socket
import threading
from io import StringIO

import pytest

from app.cli import (
    _DemoServer,
    format_cycle_output,
    print_header,
    run_demo,
)
from core.cycle import CycleResult, run_monitoring_cycle
from core.failover import PathState, PathTransition, TransitionType
from core.monitor import MonitorResult, PathMetrics
from core.path import Path
from core.scoring import ScoredPath, ScoringResult, ScoringWeights

# Also reuse the builder helpers from test_failover
from core.loss import ProbeStats
from core.probe import ProbeStatus
from core.stats import RttStats
from core.throughput import ThroughputResult


# ── Minimal object builders ────────────────────────────────────────────────────


def _make_path(name: str) -> Path:
    return Path(name=name, host="127.0.0.1", port=9000)


def _make_probe_stats(name: str, loss_rate: float = 0.0) -> ProbeStats:
    successful = 1 if loss_rate < 100.0 else 0
    return ProbeStats(
        host="127.0.0.1", port=9000,
        total=1, successful=successful, failed=1 - successful,
        loss_rate=loss_rate,
        rtt_values_ms=(10.0,) if successful else (),
        raw_results=(),
    )


def _make_rtt_stats(mean_ms: float = 10.0, jitter_ms: float = 1.0) -> RttStats:
    return RttStats(count=1, mean_ms=mean_ms, min_ms=mean_ms, max_ms=mean_ms, jitter_ms=jitter_ms)


def _make_throughput(mbps: float | None = 80.0) -> ThroughputResult:
    bps = mbps * 1_000_000 / 8 if mbps is not None else None
    return ThroughputResult(
        host="127.0.0.1", port=9000,
        status=ProbeStatus.SUCCESS if mbps is not None else ProbeStatus.REFUSED,
        bytes_transferred=1024 if mbps is not None else None,
        elapsed_seconds=0.01 if mbps is not None else None,
        throughput_bps=bps, throughput_mbps=mbps,
        error_message=None if mbps is not None else "refused",
    )


def _make_path_metrics(
    name: str,
    rtt_ms: float = 10.0,
    jitter_ms: float = 1.0,
    loss_rate: float = 0.0,
    throughput_mbps: float | None = 80.0,
) -> PathMetrics:
    path = _make_path(name)
    available = loss_rate < 100.0
    return PathMetrics(
        path=path,
        probe_stats=_make_probe_stats(name, loss_rate),
        rtt_stats=_make_rtt_stats(rtt_ms, jitter_ms) if available else RttStats(0, None, None, None, None),
        throughput=_make_throughput(throughput_mbps if available else None),
    )


def _make_scored_path(
    name: str,
    available: bool = True,
    total_score: float = 75.0,
    rank: int = 1,
) -> ScoredPath:
    pm = _make_path_metrics(
        name,
        loss_rate=0.0 if available else 100.0,
    )
    return ScoredPath(
        path=pm.path, metrics=pm, available=available,
        rtt_score=total_score, loss_score=total_score,
        jitter_score=total_score, throughput_score=total_score,
        total_score=total_score, rank=rank,
    )


def _make_cycle_result(
    names: list[str],
    preferred_name: str | None,
    event_type: TransitionType,
    previous_name: str | None = None,
) -> CycleResult:
    """Build a minimal CycleResult for formatting tests — no network needed."""
    scored_paths = [
        _make_scored_path(n, available=(n == preferred_name or n not in (previous_name,)),
                          rank=i + 1)
        for i, n in enumerate(names)
    ]
    preferred_sp = next((sp for sp in scored_paths if sp.path.name == preferred_name), None)
    monitor_metrics = {
        n: _make_path_metrics(n)
        for n in names
    }
    transition = PathTransition(
        event_type=event_type,
        previous_path=_make_path(previous_name) if previous_name else None,
        new_path=_make_path(preferred_name) if preferred_name else None,
        previous_scored_path=None,
        new_scored_path=preferred_sp,
    )
    return CycleResult(
        monitoring_result=MonitorResult(metrics=monitor_metrics),
        scoring_result=ScoringResult(
            scored_paths=scored_paths,
            preferred_path=preferred_sp,
            weights=ScoringWeights(),
        ),
        transition=transition,
    )


# ── Shared live server fixture ─────────────────────────────────────────────────

_FAST = dict(probe_count=1, probe_timeout=0.5, transfer_bytes=4096, throughput_timeout=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Module imports and basic structure
# ══════════════════════════════════════════════════════════════════════════════


class TestImports:
    def test_cli_imports_without_error(self) -> None:
        import app.cli  # noqa: F401

    def test_format_cycle_output_is_callable(self) -> None:
        assert callable(format_cycle_output)

    def test_print_header_is_callable(self) -> None:
        assert callable(print_header)

    def test_run_demo_is_coroutine_function(self) -> None:
        import asyncio
        assert asyncio.iscoroutinefunction(run_demo)


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Header output
# ══════════════════════════════════════════════════════════════════════════════


class TestHeaderOutput:
    def test_header_contains_netmon_text(self) -> None:
        out = StringIO()
        print_header(out)
        text = out.getvalue()
        assert "NETMON" in text.upper() or "Network Traffic Monitor" in text

    def test_header_contains_demo_disclaimer(self) -> None:
        out = StringIO()
        print_header(out)
        text = out.getvalue()
        assert "demo" in text.lower() or "loopback" in text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: format_cycle_output — path names
# ══════════════════════════════════════════════════════════════════════════════


class TestPathNamesInOutput:
    def test_primary_path_name_displayed(self) -> None:
        result = _make_cycle_result(
            ["primary", "backup"], "primary", TransitionType.INITIAL_SELECTION
        )
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "primary" in out.getvalue()

    def test_backup_path_name_displayed(self) -> None:
        result = _make_cycle_result(
            ["primary", "backup"], "primary", TransitionType.INITIAL_SELECTION
        )
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "backup" in out.getvalue()

    def test_single_path_name_displayed(self) -> None:
        result = _make_cycle_result(["only"], "only", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "only" in out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: format_cycle_output — metric values from CycleResult
# ══════════════════════════════════════════════════════════════════════════════


class TestMetricValuesDisplayed:
    """
    Verify that RTT, loss, jitter, throughput values from the monitoring
    result appear in the formatted output.

    We inject specific metric values and check the output contains them.
    We do NOT assert exact strings because ANSI codes may wrap values.
    We check for the numeric content (e.g., "10.0" or "ms").
    """

    def test_rtt_unit_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "ms" in out.getvalue()

    def test_loss_percentage_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "%" in out.getvalue()

    def test_throughput_unit_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "Mbps" in out.getvalue()

    def test_rtt_numeric_value_displayed(self) -> None:
        """RTT of 10.0ms → '10.0' appears in output."""
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "10.0" in out.getvalue()

    def test_score_numeric_value_displayed(self) -> None:
        """Score of 75.0 → '75.0' appears in output."""
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "75.0" in out.getvalue()

    def test_cycle_number_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(3, result, out)
        assert "3" in out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Preferred path display
# ══════════════════════════════════════════════════════════════════════════════


class TestPreferredPathDisplay:
    def test_preferred_path_label_present(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "Preferred Path" in out.getvalue()

    def test_preferred_path_name_in_summary(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        # "primary" appears both in the table row AND in "Preferred Path : primary"
        assert out.getvalue().count("primary") >= 2

    def test_preferred_path_none_when_no_available(self) -> None:
        result = _make_cycle_result(["dead"], None, TransitionType.NO_AVAILABLE_PATH)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "none" in out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Event type labels
# ══════════════════════════════════════════════════════════════════════════════


class TestEventLabels:
    def test_initial_selection_label_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "INITIAL_SELECTION" in out.getvalue()

    def test_no_change_label_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.NO_CHANGE,
                                     previous_name="primary")
        out = StringIO()
        format_cycle_output(2, result, out)
        assert "NO_CHANGE" in out.getvalue()

    def test_failover_label_displayed(self) -> None:
        result = _make_cycle_result(["primary", "backup"], "backup", TransitionType.FAILOVER,
                                     previous_name="primary")
        out = StringIO()
        format_cycle_output(3, result, out)
        assert "FAILOVER" in out.getvalue()

    def test_no_available_path_label_displayed(self) -> None:
        result = _make_cycle_result(["dead"], None, TransitionType.NO_AVAILABLE_PATH)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "NO_AVAILABLE_PATH" in out.getvalue()

    def test_event_label_displayed(self) -> None:
        result = _make_cycle_result(["primary"], "primary", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "Event" in out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Failover transition arrow
# ══════════════════════════════════════════════════════════════════════════════


class TestFailoverTransitionArrow:
    def test_arrow_displayed_on_failover(self) -> None:
        result = _make_cycle_result(
            ["primary", "backup"], "backup", TransitionType.FAILOVER,
            previous_name="primary",
        )
        out = StringIO()
        format_cycle_output(3, result, out)
        text = out.getvalue()
        assert "→" in text or "->" in text

    def test_previous_path_name_on_failover(self) -> None:
        result = _make_cycle_result(
            ["primary", "backup"], "backup", TransitionType.FAILOVER,
            previous_name="primary",
        )
        out = StringIO()
        format_cycle_output(3, result, out)
        assert "primary" in out.getvalue()

    def test_new_path_name_on_failover(self) -> None:
        result = _make_cycle_result(
            ["primary", "backup"], "backup", TransitionType.FAILOVER,
            previous_name="primary",
        )
        out = StringIO()
        format_cycle_output(3, result, out)
        assert "backup" in out.getvalue()

    def test_transition_line_only_on_failover(self) -> None:
        """'Transition' line must NOT appear for NO_CHANGE."""
        result = _make_cycle_result(
            ["primary"], "primary", TransitionType.NO_CHANGE,
            previous_name="primary",
        )
        out = StringIO()
        format_cycle_output(2, result, out)
        assert "Transition" not in out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: Unavailable path display
# ══════════════════════════════════════════════════════════════════════════════


class TestUnavailableDisplay:
    def test_unavailable_label_shown_for_dead_path(self) -> None:
        """
        Build a CycleResult where 'dead' path is unavailable.
        The output must contain 'UNAVAILABLE'.
        """
        dead_pm = _make_path_metrics("dead", loss_rate=100.0)
        dead_sp = _make_scored_path("dead", available=False, total_score=0.0, rank=2)

        live_pm = _make_path_metrics("live")
        live_sp = _make_scored_path("live", available=True, total_score=75.0, rank=1)

        transition = PathTransition(
            event_type=TransitionType.FAILOVER,
            previous_path=_make_path("dead"),
            new_path=_make_path("live"),
            previous_scored_path=None,
            new_scored_path=live_sp,
        )
        result = CycleResult(
            monitoring_result=MonitorResult(metrics={"dead": dead_pm, "live": live_pm}),
            scoring_result=ScoringResult(
                scored_paths=[live_sp, dead_sp],
                preferred_path=live_sp,
                weights=ScoringWeights(),
            ),
            transition=transition,
        )
        out = StringIO()
        format_cycle_output(3, result, out)
        assert "UNAVAILABLE" in out.getvalue()

    def test_available_status_shown_for_live_path(self) -> None:
        result = _make_cycle_result(["live"], "live", TransitionType.INITIAL_SELECTION)
        out = StringIO()
        format_cycle_output(1, result, out)
        assert "available" in out.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Part 9: Full demo run (live servers)
# ══════════════════════════════════════════════════════════════════════════════


class TestFullDemoRun:
    """Run the complete demo against real local servers and verify the output."""

    @pytest.mark.asyncio
    async def test_demo_completes_without_error(self) -> None:
        out = StringIO()
        await run_demo(out)  # must not raise

    @pytest.mark.asyncio
    async def test_demo_output_contains_cycle_numbers(self) -> None:
        out = StringIO()
        await run_demo(out)
        text = out.getvalue()
        assert "Cycle 1" in text
        assert "Cycle 2" in text
        assert "Cycle 3" in text
        assert "Cycle 4" in text

    @pytest.mark.asyncio
    async def test_demo_output_contains_initial_selection(self) -> None:
        out = StringIO()
        await run_demo(out)
        assert "INITIAL_SELECTION" in out.getvalue()

    @pytest.mark.asyncio
    async def test_demo_output_contains_no_change(self) -> None:
        out = StringIO()
        await run_demo(out)
        assert "NO_CHANGE" in out.getvalue()

    @pytest.mark.asyncio
    async def test_demo_output_contains_failover(self) -> None:
        out = StringIO()
        await run_demo(out)
        assert "FAILOVER" in out.getvalue()

    @pytest.mark.asyncio
    async def test_demo_output_contains_path_names(self) -> None:
        out = StringIO()
        await run_demo(out)
        text = out.getvalue()
        assert "primary" in text
        assert "backup" in text

    @pytest.mark.asyncio
    async def test_demo_output_contains_preferred_path_label(self) -> None:
        out = StringIO()
        await run_demo(out)
        assert "Preferred Path" in out.getvalue()

    @pytest.mark.asyncio
    async def test_demo_output_contains_mbps(self) -> None:
        out = StringIO()
        await run_demo(out)
        assert "Mbps" in out.getvalue()

    @pytest.mark.asyncio
    async def test_demo_output_contains_ms(self) -> None:
        out = StringIO()
        await run_demo(out)
        assert "ms" in out.getvalue()

    @pytest.mark.asyncio
    async def test_demo_output_does_not_depend_on_internet(self) -> None:
        """
        The demo uses local loopback servers — no internet access needed.
        Verify that the output explicitly mentions 'loopback' or 'local'.
        """
        out = StringIO()
        await run_demo(out)
        text = out.getvalue().lower()
        assert "loopback" in text or "local" in text or "127.0.0.1" in text


# ══════════════════════════════════════════════════════════════════════════════
# Part 10: _DemoServer behaviour
# ══════════════════════════════════════════════════════════════════════════════


class TestDemoServer:
    def test_demo_server_starts_and_assigns_port(self) -> None:
        srv = _DemoServer("test")
        srv.start()
        assert srv.port > 0
        assert srv.host == "127.0.0.1"
        srv.stop()

    def test_demo_server_path_returns_path_object(self) -> None:
        srv = _DemoServer("mypath")
        srv.start()
        p = srv.path
        assert isinstance(p, Path)
        assert p.name == "mypath"
        srv.stop()

    def test_demo_server_responds_to_ping(self) -> None:
        srv = _DemoServer("test")
        srv.start()
        try:
            with socket.create_connection((srv.host, srv.port), timeout=1.0) as conn:
                conn.sendall(b"PING\n")
                resp = conn.recv(16)
                assert resp == b"PONG\n"
        finally:
            srv.stop()

    def test_demo_server_stop_prevents_connections(self) -> None:
        """After stop(), connecting should fail (refused or timeout)."""
        srv = _DemoServer("test")
        srv.start()
        host, port = srv.host, srv.port
        srv.stop()
        import time
        time.sleep(0.05)  # let OS fully close the socket
        with pytest.raises(OSError):
            socket.create_connection((host, port), timeout=0.3)

    @pytest.mark.asyncio
    async def test_demo_server_can_be_monitored(self) -> None:
        """A _DemoServer responds correctly to the monitoring pipeline."""
        srv = _DemoServer("srv")
        srv.start()
        try:
            state = PathState()
            result = await run_monitoring_cycle(
                [srv.path], state,
                probe_count=1, probe_timeout=0.5,
                transfer_bytes=4096, throughput_timeout=2.0,
            )
            pm = result.monitoring_result.metrics["srv"]
            assert pm.probe_stats.loss_rate == 0.0
        finally:
            srv.stop()
