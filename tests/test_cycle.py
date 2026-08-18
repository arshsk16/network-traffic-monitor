"""
tests/test_cycle.py — Integration tests for core/cycle.py.

Test strategy
─────────────
These are true integration tests: they run run_monitoring_cycle() against
real controlled local TCP servers. This exercises the complete pipeline:

    local dual server → monitor_paths() → score_paths() → state.update()
                                                       → CycleResult

All servers are local loopback (127.0.0.1) with ephemeral ports selected
by the OS. No internet connectivity is required. No external services.

For tests involving state transitions without real monitoring, we use the
scoring/failover layers directly to construct controlled ScoringResult
objects — exactly as the test_scoring.py and test_failover.py tests do.
This avoids making transition tests depend on real network timing.

Server infrastructure
─────────────────────
We reuse the _make_dual_server() approach from test_monitor.py:
  - Combined PING→PONG + SIZE:N→data→DONE handler on one port
  - Threaded accept loop with stop_event for clean teardown
  - Data is read through the buffered reader to avoid the conn.recv() bug

Each test that needs a live server creates one inside the test or uses
a pytest fixture. Servers are torn down after each test.

Async tests
───────────
run_monitoring_cycle() is async. We use pytest-asyncio with
@pytest.mark.asyncio (STRICT mode, per pyproject.toml configuration).

What is tested
──────────────
Part 1: CycleResult structure
Part 2: Single complete cycle (live server)
Part 3: Multi-path complete cycle (live servers)
Part 4: State transition across cycles (live servers)
Part 5: Failover detection (one path dies between cycles)
Part 6: NO_AVAILABLE_PATH when all paths are down
Part 7: State persistence across cycles
Part 8: CycleResult convenience properties
Part 9: Custom weights flow through to CycleResult
Part 10: Empty paths list
"""

from __future__ import annotations

import socket
import threading
from asyncio import run

import pytest
import pytest_asyncio

from core.cycle import CycleResult, run_monitoring_cycle
from core.failover import PathState, TransitionType
from core.path import Path
from core.scoring import ScoringWeights


# ── Shared server infrastructure ───────────────────────────────────────────────


def _make_dual_server(stop_event: threading.Event):
    """
    Create a combined echo + throughput sink server.

    Handles:
      - b"PING\\n"   → b"PONG\\n"         (RTT probe)
      - b"SIZE:N\\n" → read N bytes → b"DONE\\n"  (throughput)

    Data is always read through the buffered reader (not raw conn.recv())
    to avoid pre-buffered bytes being missed.

    Returns (server_sock, host, port).
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(64)
    server_sock.settimeout(0.05)
    host, port = server_sock.getsockname()

    def _handle(conn: socket.socket) -> None:
        try:
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                reader = conn.makefile("rb")
                first_line = reader.readline()
                if first_line == b"PING\n":
                    conn.sendall(b"PONG\n")
                elif first_line.startswith(b"SIZE:"):
                    n_bytes = int(first_line.split(b":")[1].strip())
                    remaining = n_bytes
                    while remaining > 0:
                        chunk = reader.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                    conn.sendall(b"DONE\n")
        except OSError:
            pass

    def _accept_loop() -> None:
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                threading.Thread(target=_handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    threading.Thread(target=_accept_loop, daemon=True).start()
    return server_sock, host, port


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def live_server():
    """
    One live dual server. Yields (host, port).
    Cleaned up after each test.
    """
    stop = threading.Event()
    sock, host, port = _make_dual_server(stop)
    yield host, port
    stop.set()
    sock.close()


@pytest.fixture()
def live_path(live_server):
    """A Path pointing at the live_server fixture."""
    host, port = live_server
    return Path(name="primary", host=host, port=port)


@pytest.fixture()
def two_live_servers():
    """
    Two independent dual servers on different ports.
    Yields [(host_a, port_a), (host_b, port_b)].
    """
    stop_a = threading.Event()
    stop_b = threading.Event()
    sock_a, host_a, port_a = _make_dual_server(stop_a)
    sock_b, host_b, port_b = _make_dual_server(stop_b)
    yield [(host_a, port_a), (host_b, port_b)]
    stop_a.set()
    stop_b.set()
    sock_a.close()
    sock_b.close()


def _dead_path(name: str = "dead") -> Path:
    """
    Return a Path whose port has no server.

    Binds and immediately closes a socket so the OS recycles the port
    number. Connecting to this path will be refused immediately.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    return Path(name=name, host="127.0.0.1", port=port)


# ── Fast monitoring config (keeps tests quick) ─────────────────────────────────

_FAST = dict(
    probe_count=1,
    probe_timeout=0.5,
    transfer_bytes=4096,
    throughput_timeout=2.0,
)


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: CycleResult structure (no network needed)
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleResultStructure:
    """CycleResult type structure and convenience properties."""

    @pytest.mark.asyncio
    async def test_returns_cycle_result(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert isinstance(result, CycleResult)

    @pytest.mark.asyncio
    async def test_cycle_result_has_monitoring_result(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        from core.monitor import MonitorResult
        assert isinstance(result.monitoring_result, MonitorResult)

    @pytest.mark.asyncio
    async def test_cycle_result_has_scoring_result(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        from core.scoring import ScoringResult
        assert isinstance(result.scoring_result, ScoringResult)

    @pytest.mark.asyncio
    async def test_cycle_result_has_transition(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        from core.failover import PathTransition
        assert isinstance(result.transition, PathTransition)

    @pytest.mark.asyncio
    async def test_cycle_result_is_immutable(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        with pytest.raises(Exception):
            result.transition = None  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_str_does_not_raise(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        s = str(result)
        assert isinstance(s, str)
        assert len(s) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Single complete cycle
# ══════════════════════════════════════════════════════════════════════════════


class TestSingleCompleteCycle:
    """One live path → one complete cycle → expected outputs."""

    @pytest.mark.asyncio
    async def test_monitoring_result_contains_path(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert "primary" in result.monitoring_result.metrics

    @pytest.mark.asyncio
    async def test_monitoring_result_probe_stats_not_none(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        pm = result.monitoring_result.metrics["primary"]
        assert pm.probe_stats is not None

    @pytest.mark.asyncio
    async def test_monitoring_result_zero_loss_on_live_path(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        pm = result.monitoring_result.metrics["primary"]
        assert pm.probe_stats.loss_rate == 0.0

    @pytest.mark.asyncio
    async def test_scoring_result_contains_path(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        names = {sp.path.name for sp in result.scoring_result.scored_paths}
        assert "primary" in names

    @pytest.mark.asyncio
    async def test_preferred_path_is_primary(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.scoring_result.preferred_path is not None
        assert result.scoring_result.preferred_path.path.name == "primary"

    @pytest.mark.asyncio
    async def test_first_cycle_is_initial_selection(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.transition.event_type == TransitionType.INITIAL_SELECTION

    @pytest.mark.asyncio
    async def test_preferred_path_name_shortcut(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.preferred_path_name == "primary"

    @pytest.mark.asyncio
    async def test_first_cycle_not_failover(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.is_failover is False

    @pytest.mark.asyncio
    async def test_first_cycle_is_change(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.is_change is True


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Multi-path cycle
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiPathCycle:
    """Two live paths → both appear in monitoring and scoring results."""

    @pytest.mark.asyncio
    async def test_both_paths_in_monitoring_result(
        self, two_live_servers: list
    ) -> None:
        server_infos = list(enumerate(two_live_servers))
        paths = [
            Path(name="alpha" if i == 0 else "beta", host=h, port=p)
            for i, (h, p) in server_infos
        ]
        state = PathState()
        result = await run_monitoring_cycle(paths, state, **_FAST)
        assert "alpha" in result.monitoring_result.metrics
        assert "beta" in result.monitoring_result.metrics
        assert len(result.monitoring_result.metrics) == 2

    @pytest.mark.asyncio
    async def test_both_paths_in_scoring_result(
        self, two_live_servers: list
    ) -> None:
        paths = [
            Path(name=f"path-{i}", host=h, port=p)
            for i, (h, p) in enumerate(two_live_servers)
        ]
        state = PathState()
        result = await run_monitoring_cycle(paths, state, **_FAST)
        names = {sp.path.name for sp in result.scoring_result.scored_paths}
        assert "path-0" in names
        assert "path-1" in names

    @pytest.mark.asyncio
    async def test_preferred_path_is_not_none_with_two_live_paths(
        self, two_live_servers: list
    ) -> None:
        paths = [
            Path(name=f"p{i}", host=h, port=p)
            for i, (h, p) in enumerate(two_live_servers)
        ]
        state = PathState()
        result = await run_monitoring_cycle(paths, state, **_FAST)
        assert result.scoring_result.preferred_path is not None

    @pytest.mark.asyncio
    async def test_multi_path_monitoring_flows_into_scoring(
        self, two_live_servers: list
    ) -> None:
        """Scoring's scored_paths count must match monitoring's metrics count."""
        paths = [
            Path(name=f"p{i}", host=h, port=p)
            for i, (h, p) in enumerate(two_live_servers)
        ]
        state = PathState()
        result = await run_monitoring_cycle(paths, state, **_FAST)
        assert len(result.scoring_result.scored_paths) == len(
            result.monitoring_result.metrics
        )


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: State transitions across cycles
# ══════════════════════════════════════════════════════════════════════════════


class TestStateTransitionsAcrossCycles:
    """The SAME PathState instance must track state across multiple cycles."""

    @pytest.mark.asyncio
    async def test_second_healthy_cycle_is_no_change(self, live_path: Path) -> None:
        state = PathState()
        await run_monitoring_cycle([live_path], state, **_FAST)   # Cycle 1
        result2 = await run_monitoring_cycle([live_path], state, **_FAST)  # Cycle 2
        assert result2.transition.event_type == TransitionType.NO_CHANGE

    @pytest.mark.asyncio
    async def test_repeated_healthy_cycles_never_failover(self, live_path: Path) -> None:
        state = PathState()
        for _ in range(4):
            result = await run_monitoring_cycle([live_path], state, **_FAST)
        # After the 4th cycle, must still be NO_CHANGE
        assert result.transition.event_type == TransitionType.NO_CHANGE

    @pytest.mark.asyncio
    async def test_state_reflects_current_preferred_after_cycle(
        self, live_path: Path
    ) -> None:
        state = PathState()
        await run_monitoring_cycle([live_path], state, **_FAST)
        assert state.current_preferred_path is not None
        assert state.current_preferred_path.name == "primary"

    @pytest.mark.asyncio
    async def test_second_cycle_previous_path_is_primary(self, live_path: Path) -> None:
        state = PathState()
        await run_monitoring_cycle([live_path], state, **_FAST)
        result2 = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result2.transition.previous_path is not None
        assert result2.transition.previous_path.name == "primary"


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Failover detection
# ══════════════════════════════════════════════════════════════════════════════


class TestFailoverDetection:
    """Simulate primary path going down between cycles → FAILOVER."""

    @pytest.mark.asyncio
    async def test_failover_when_primary_dies(
        self, two_live_servers: list
    ) -> None:
        """
        Cycle 1: both paths alive → primary preferred (alphabetically "p0").
        Cycle 2: p0 dead, p1 alive → FAILOVER to p1.

        We control this by:
         - Cycle 1: use path p0 (live) + path p1 (live)
         - Cycle 2: replace p0 with a dead path but keep same name
        """
        (h0, port0), (h1, port1) = two_live_servers
        paths_cycle1 = [
            Path(name="p0", host=h0, port=port0),
            Path(name="p1", host=h1, port=port1),
        ]

        state = PathState()
        result1 = await run_monitoring_cycle(paths_cycle1, state, **_FAST)
        # Whatever was selected in cycle 1, now make that path dead
        preferred_after_cycle1 = result1.preferred_path_name

        # Cycle 2: replace the preferred path with a dead path
        dead = _dead_path(preferred_after_cycle1)
        # Pick the other path name
        other_name = "p1" if preferred_after_cycle1 == "p0" else "p0"
        other_host, other_port = (h1, port1) if other_name == "p1" else (h0, port0)
        live_other = Path(name=other_name, host=other_host, port=other_port)

        paths_cycle2 = [dead, live_other]
        result2 = await run_monitoring_cycle(paths_cycle2, state, **_FAST)

        assert result2.transition.event_type == TransitionType.FAILOVER
        assert result2.is_failover is True
        assert result2.preferred_path_name == other_name

    @pytest.mark.asyncio
    async def test_failover_transition_contains_both_paths(
        self, two_live_servers: list
    ) -> None:
        """
        After a failover, transition.previous_path is the old path and
        transition.new_path is the new path.
        """
        (h0, port0), (h1, port1) = two_live_servers

        state = PathState()
        # Cycle 1: only p0 available → INITIAL_SELECTION
        paths_c1 = [Path(name="p0", host=h0, port=port0)]
        r1 = await run_monitoring_cycle(paths_c1, state, **_FAST)
        assert r1.transition.event_type == TransitionType.INITIAL_SELECTION

        # Cycle 2: p0 dead, p1 alive → FAILOVER
        dead_p0 = _dead_path("p0")
        live_p1 = Path(name="p1", host=h1, port=port1)
        r2 = await run_monitoring_cycle([dead_p0, live_p1], state, **_FAST)

        assert r2.transition.event_type == TransitionType.FAILOVER
        assert r2.transition.previous_path.name == "p0"
        assert r2.transition.new_path.name == "p1"


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: NO_AVAILABLE_PATH
# ══════════════════════════════════════════════════════════════════════════════


class TestNoAvailablePath:
    """All paths dead → NO_AVAILABLE_PATH."""

    @pytest.mark.asyncio
    async def test_dead_path_produces_no_available_path(self) -> None:
        dead = _dead_path("dead")
        state = PathState()
        result = await run_monitoring_cycle(
            [dead], state,
            probe_count=1, probe_timeout=0.3,
            transfer_bytes=4096, throughput_timeout=0.3,
        )
        assert result.transition.event_type == TransitionType.NO_AVAILABLE_PATH

    @pytest.mark.asyncio
    async def test_preferred_path_name_is_none_when_all_dead(self) -> None:
        dead = _dead_path("dead")
        state = PathState()
        result = await run_monitoring_cycle(
            [dead], state,
            probe_count=1, probe_timeout=0.3,
            transfer_bytes=4096, throughput_timeout=0.3,
        )
        assert result.preferred_path_name is None

    @pytest.mark.asyncio
    async def test_state_preferred_path_is_none_when_all_dead(self) -> None:
        dead = _dead_path("dead")
        state = PathState()
        await run_monitoring_cycle(
            [dead], state,
            probe_count=1, probe_timeout=0.3,
            transfer_bytes=4096, throughput_timeout=0.3,
        )
        assert state.current_preferred_path is None

    @pytest.mark.asyncio
    async def test_no_available_is_a_change(self) -> None:
        dead = _dead_path("dead")
        state = PathState()
        result = await run_monitoring_cycle(
            [dead], state,
            probe_count=1, probe_timeout=0.3,
            transfer_bytes=4096, throughput_timeout=0.3,
        )
        assert result.is_change is True

    @pytest.mark.asyncio
    async def test_transition_from_live_to_all_dead(self, live_path: Path) -> None:
        state = PathState()
        # Cycle 1: live → INITIAL_SELECTION
        await run_monitoring_cycle([live_path], state, **_FAST)

        # Cycle 2: all dead → NO_AVAILABLE_PATH
        dead = _dead_path("primary")  # same name, no server
        result = await run_monitoring_cycle(
            [dead], state,
            probe_count=1, probe_timeout=0.3,
            transfer_bytes=4096, throughput_timeout=0.3,
        )
        assert result.transition.event_type == TransitionType.NO_AVAILABLE_PATH
        assert result.transition.previous_path.name == "primary"
        assert result.transition.new_path is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: State persists across cycles
# ══════════════════════════════════════════════════════════════════════════════


class TestStatePersistenceAcrossCycles:
    """PathState must accumulate correctly over N cycles."""

    @pytest.mark.asyncio
    async def test_state_persists_same_instance_required(
        self, live_path: Path
    ) -> None:
        """
        Using a DIFFERENT PathState for each cycle would always produce
        INITIAL_SELECTION. Using the SAME one produces NO_CHANGE on cycle 2.
        Demonstrates why state reuse is essential.
        """
        state_shared = PathState()
        # Cycle 1
        await run_monitoring_cycle([live_path], state_shared, **_FAST)
        # Cycle 2 — same state → NO_CHANGE
        r2 = await run_monitoring_cycle([live_path], state_shared, **_FAST)
        assert r2.transition.event_type == TransitionType.NO_CHANGE

        # Now compare: fresh state for cycle 2 → INITIAL_SELECTION again
        state_fresh = PathState()
        r_fresh = await run_monitoring_cycle([live_path], state_fresh, **_FAST)
        assert r_fresh.transition.event_type == TransitionType.INITIAL_SELECTION

    @pytest.mark.asyncio
    async def test_multiple_cycles_accumulate_correctly(
        self, live_path: Path
    ) -> None:
        """
        5 cycles with the same live path.
        Cycle 1: INITIAL_SELECTION
        Cycles 2-5: NO_CHANGE
        """
        state = PathState()
        results = []
        for _ in range(5):
            results.append(
                await run_monitoring_cycle([live_path], state, **_FAST)
            )

        assert results[0].transition.event_type == TransitionType.INITIAL_SELECTION
        for r in results[1:]:
            assert r.transition.event_type == TransitionType.NO_CHANGE


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: CycleResult convenience properties
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleResultConvenienceProperties:
    """Shortcut properties on CycleResult must reflect the underlying data."""

    @pytest.mark.asyncio
    async def test_is_failover_false_on_initial_selection(
        self, live_path: Path
    ) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.is_failover is False

    @pytest.mark.asyncio
    async def test_is_failover_false_on_no_change(self, live_path: Path) -> None:
        state = PathState()
        await run_monitoring_cycle([live_path], state, **_FAST)
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.is_failover is False

    @pytest.mark.asyncio
    async def test_is_change_true_on_initial_selection(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.is_change is True

    @pytest.mark.asyncio
    async def test_is_change_false_on_no_change(self, live_path: Path) -> None:
        state = PathState()
        await run_monitoring_cycle([live_path], state, **_FAST)
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.is_change is False

    @pytest.mark.asyncio
    async def test_preferred_path_name_matches_transition_new_path(
        self, live_path: Path
    ) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        if result.transition.new_path:
            assert result.preferred_path_name == result.transition.new_path.name
        else:
            assert result.preferred_path_name is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 9: Custom weights flow through
# ══════════════════════════════════════════════════════════════════════════════


class TestCustomWeights:
    """Custom ScoringWeights passed to run_monitoring_cycle must appear in result."""

    @pytest.mark.asyncio
    async def test_custom_weights_stored_in_scoring_result(
        self, live_path: Path
    ) -> None:
        weights = ScoringWeights(rtt=0.50, loss=0.25, jitter=0.15, throughput=0.10)
        state = PathState()
        result = await run_monitoring_cycle(
            [live_path], state, weights=weights, **_FAST
        )
        assert result.scoring_result.weights.rtt == pytest.approx(0.50)
        assert result.scoring_result.weights.loss == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_default_weights_when_none_passed(self, live_path: Path) -> None:
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        assert result.scoring_result.weights.rtt == pytest.approx(0.30)
        assert result.scoring_result.weights.loss == pytest.approx(0.30)


# ══════════════════════════════════════════════════════════════════════════════
# Part 10: Empty paths list
# ══════════════════════════════════════════════════════════════════════════════


class TestEmptyPathsList:
    """An empty paths list must return a valid CycleResult cleanly."""

    @pytest.mark.asyncio
    async def test_empty_paths_returns_cycle_result(self) -> None:
        state = PathState()
        result = await run_monitoring_cycle([], state, **_FAST)
        assert isinstance(result, CycleResult)

    @pytest.mark.asyncio
    async def test_empty_paths_monitoring_result_is_empty(self) -> None:
        state = PathState()
        result = await run_monitoring_cycle([], state, **_FAST)
        assert result.monitoring_result.metrics == {}

    @pytest.mark.asyncio
    async def test_empty_paths_scoring_result_has_no_paths(self) -> None:
        state = PathState()
        result = await run_monitoring_cycle([], state, **_FAST)
        assert result.scoring_result.scored_paths == []

    @pytest.mark.asyncio
    async def test_empty_paths_preferred_is_none(self) -> None:
        state = PathState()
        result = await run_monitoring_cycle([], state, **_FAST)
        assert result.preferred_path_name is None

    @pytest.mark.asyncio
    async def test_empty_paths_transition_is_no_available_path(self) -> None:
        state = PathState()
        result = await run_monitoring_cycle([], state, **_FAST)
        assert result.transition.event_type == TransitionType.NO_AVAILABLE_PATH


# ══════════════════════════════════════════════════════════════════════════════
# Part 11: Data flows — monitoring feeds scoring, scoring feeds state
# ══════════════════════════════════════════════════════════════════════════════


class TestDataFlows:
    """Verify that each layer's output flows into the next correctly."""

    @pytest.mark.asyncio
    async def test_monitoring_metrics_count_matches_paths_count(
        self, two_live_servers: list
    ) -> None:
        paths = [
            Path(name=f"p{i}", host=h, port=p)
            for i, (h, p) in enumerate(two_live_servers)
        ]
        state = PathState()
        result = await run_monitoring_cycle(paths, state, **_FAST)
        assert len(result.monitoring_result.metrics) == len(paths)

    @pytest.mark.asyncio
    async def test_scored_paths_count_matches_monitoring_metrics(
        self, two_live_servers: list
    ) -> None:
        paths = [
            Path(name=f"p{i}", host=h, port=p)
            for i, (h, p) in enumerate(two_live_servers)
        ]
        state = PathState()
        result = await run_monitoring_cycle(paths, state, **_FAST)
        assert len(result.scoring_result.scored_paths) == len(
            result.monitoring_result.metrics
        )

    @pytest.mark.asyncio
    async def test_transition_new_path_matches_scoring_preferred(
        self, live_path: Path
    ) -> None:
        """transition.new_path must be the same path as scoring_result.preferred_path."""
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        if result.scoring_result.preferred_path is not None:
            assert result.transition.new_path is not None
            assert (
                result.transition.new_path.name
                == result.scoring_result.preferred_path.path.name
            )

    @pytest.mark.asyncio
    async def test_state_preferred_matches_cycle_preferred(
        self, live_path: Path
    ) -> None:
        """After update(), state.current_preferred_path matches the cycle result."""
        state = PathState()
        result = await run_monitoring_cycle([live_path], state, **_FAST)
        if result.preferred_path_name:
            assert state.current_preferred_path is not None
            assert state.current_preferred_path.name == result.preferred_path_name
        else:
            assert state.current_preferred_path is None
