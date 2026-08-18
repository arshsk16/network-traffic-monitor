"""
tests/test_monitor.py — Tests for core/monitor.py and core/path.py.

Test strategy
─────────────
All tests use controlled local TCP servers. No internet hosts, no external
dependencies, no Cloudflare, no Google, no real SD-WAN appliances.

A "path" in these tests is simply a named (host, port) pointing to a locally
spun-up TCP server. This is sufficient to validate the monitoring logic: the
monitoring layer only cares about network addresses, not physical topology.

What we test
────────────
 1. Path model: construction, validation, immutability
 2. One configured path can be monitored (happy path)
 3. Multiple configured paths can be monitored
 4. Results retain the correct path identity
 5. Multiple paths are monitored concurrently (deterministic synchronization)
 6. A failed path does not crash monitoring of healthy paths
 7. Timeout / failure is captured in the correct path's result
 8. Empty path list is handled cleanly
 9. All existing Steps 1–6 tests continue passing (verified by running the
    full suite)

Concurrency verification strategy
───────────────────────────────────
We do NOT rely purely on elapsed-time assertions to verify concurrency.
Elapsed-time assertions are inherently flaky on slow CI machines. Instead,
we use a threading.Barrier:

    A threading.Barrier(N) blocks each of N threads until all N have arrived.
    If monitoring is truly concurrent, N path-threads will all hit the barrier
    "simultaneously" and all be released together.
    If monitoring were sequential, only one thread would run at a time and the
    barrier would never be filled (causing a timeout, which we catch and assert).

This is a deterministic synchronization test — it does not depend on wall-clock
timing, only on the observable fact that multiple threads ran concurrently.

Server fixtures
───────────────
We define server fixtures locally in this file rather than in conftest.py
because they are specific to monitor tests. conftest.py only holds fixtures
shared across multiple test modules.

Each fixture:
  - Binds a socket to 127.0.0.1:0 (OS picks a free port)
  - Runs an accept loop in a daemon thread
  - Yields (host, port) or a Path object
  - Cleans up after the test (stop_event + join)

asyncio tests
─────────────
Tests of async functions use pytest-anyio or the built-in asyncio.run() wrapper
via a synchronous test. We use a thin synchronous helper run() = asyncio.run()
to keep tests readable without requiring pytest-asyncio as a dependency.

If pytest-asyncio is available in the environment, @pytest.mark.asyncio would
be a cleaner option. We avoid it here to keep dependencies minimal.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

from core.monitor import (
    DEFAULT_PROBE_COUNT,
    MonitorResult,
    PathMetrics,
    monitor_paths,
)
from core.path import Path
from core.probe import ProbeStatus


# ── Helpers ───────────────────────────────────────────────────────────────────


def run(coro):
    """
    Run an async coroutine synchronously.

    Using asyncio.run() here instead of pytest-asyncio keeps the test file
    dependency-free. Every test in this file that tests async code goes
    through this helper.
    """
    return asyncio.run(coro)


# ── Reusable server builders ───────────────────────────────────────────────────


def _make_echo_server(stop_event: threading.Event):
    """
    Create a full TCP echo server (PING → PONG) and start its accept loop.

    Returns (server_sock, host, port). Caller is responsible for closing
    server_sock and setting stop_event to shut down the accept thread.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(32)
    server_sock.settimeout(0.05)
    host, port = server_sock.getsockname()

    def _handle(conn: socket.socket) -> None:
        try:
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                reader = conn.makefile("rb")
                if reader.readline() == b"PING\n":
                    conn.sendall(b"PONG\n")
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


def _make_throughput_sink_server(stop_event: threading.Event):
    """
    Create a TCP throughput sink server (reads SIZE header then data, sends DONE).

    Returns (server_sock, host, port).
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(32)
    server_sock.settimeout(0.05)
    host, port = server_sock.getsockname()

    def _handle(conn: socket.socket) -> None:
        try:
            with conn:
                reader = conn.makefile("rb")
                header = reader.readline().decode()
                if not header.startswith("SIZE:"):
                    return
                n_bytes = int(header.split(":")[1].strip())
                remaining = n_bytes
                while remaining > 0:
                    to_read = min(65536, remaining)
                    chunk = reader.read(to_read)
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


def _make_dual_server(stop_event: threading.Event):
    """
    Create a combined server that handles both echo (PING→PONG) and
    throughput (SIZE:N → read N bytes → DONE) on the same port.

    The protocol is distinguished by the first line received:
      - "PING\\n"  → echo mode
      - "SIZE:N\\n" → throughput sink mode

    This lets a single port serve all monitoring measurements, which is what
    monitor_paths() requires: it sends both probes and throughput to the
    same path.host:path.port.
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
                    # IMPORTANT: read data through the same buffered reader, not
                    # raw conn.recv(). The buffered reader may have already pulled
                    # some data bytes into its internal buffer when it read the
                    # header line. Switching to conn.recv() would miss those bytes,
                    # causing the recv loop to block forever waiting for bytes that
                    # were already consumed by the reader's buffer.
                    remaining = n_bytes
                    while remaining > 0:
                        to_read = min(65536, remaining)
                        chunk = reader.read(to_read)
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
def dual_server():
    """
    A combined echo + throughput sink server on a single port.

    Yields (host, port). The server handles both PING→PONG and
    SIZE:N→data→DONE on the same port, which is what monitor_paths() needs.
    """
    stop_event = threading.Event()
    server_sock, host, port = _make_dual_server(stop_event)
    yield host, port
    stop_event.set()
    server_sock.close()


@pytest.fixture()
def dual_server_path(dual_server):
    """A Path pointing at the dual_server fixture."""
    host, port = dual_server
    return Path(name="test-path", host=host, port=port)


@pytest.fixture()
def two_dual_servers():
    """
    Two independent dual servers on different ports.

    Yields list[tuple[str, int]] — [(host_a, port_a), (host_b, port_b)].
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


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Path model tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPathModel:
    """core/path.py — Path dataclass construction and validation."""

    def test_path_construction_valid(self) -> None:
        """A valid Path is created without errors."""
        p = Path(name="primary", host="10.0.0.1", port=5201)
        assert p.name == "primary"
        assert p.host == "10.0.0.1"
        assert p.port == 5201

    def test_path_is_immutable(self) -> None:
        """Frozen dataclass: attribute assignment raises FrozenInstanceError."""
        p = Path(name="primary", host="10.0.0.1", port=5201)
        with pytest.raises(Exception):  # FrozenInstanceError
            p.name = "other"  # type: ignore[misc]

    def test_path_name_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Path(name="", host="127.0.0.1", port=9000)

    def test_path_name_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Path(name="   ", host="127.0.0.1", port=9000)

    def test_path_host_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="host"):
            Path(name="x", host="", port=9000)

    def test_path_port_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="port"):
            Path(name="x", host="127.0.0.1", port=0)

    def test_path_port_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="port"):
            Path(name="x", host="127.0.0.1", port=-1)

    def test_path_port_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="port"):
            Path(name="x", host="127.0.0.1", port=65536)

    def test_path_port_boundary_valid_1(self) -> None:
        """Port 1 is the lowest valid port."""
        p = Path(name="x", host="127.0.0.1", port=1)
        assert p.port == 1

    def test_path_port_boundary_valid_65535(self) -> None:
        """Port 65535 is the highest valid port."""
        p = Path(name="x", host="127.0.0.1", port=65535)
        assert p.port == 65535

    def test_path_str_includes_name_host_port(self) -> None:
        p = Path(name="primary", host="10.0.0.1", port=5201)
        s = str(p)
        assert "primary" in s
        assert "10.0.0.1" in s
        assert "5201" in s

    def test_path_equality(self) -> None:
        """Two Paths with the same fields are equal (frozen dataclass)."""
        p1 = Path(name="x", host="127.0.0.1", port=9000)
        p2 = Path(name="x", host="127.0.0.1", port=9000)
        assert p1 == p2

    def test_path_hashable(self) -> None:
        """Frozen dataclass must be hashable (usable as dict key)."""
        p = Path(name="x", host="127.0.0.1", port=9000)
        d = {p: "value"}
        assert d[p] == "value"


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: MonitorResult and PathMetrics structural tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMonitorResultStructure:
    """Verify the shape and types of MonitorResult."""

    def test_empty_paths_returns_empty_metrics(self) -> None:
        """
        An empty path list is valid — the result has an empty metrics dict.
        """
        result = run(monitor_paths([]))
        assert isinstance(result, MonitorResult)
        assert result.metrics == {}

    def test_returns_monitor_result(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        assert isinstance(result, MonitorResult)

    def test_metrics_dict_contains_path_name(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        assert "test-path" in result.metrics

    def test_metrics_value_is_path_metrics(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        assert isinstance(result.metrics["test-path"], PathMetrics)

    def test_path_metrics_has_no_error(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        m = result.metrics["test-path"]
        assert m.error is None

    def test_path_metrics_probe_stats_not_none(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        m = result.metrics["test-path"]
        assert m.probe_stats is not None

    def test_path_metrics_rtt_stats_not_none(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        m = result.metrics["test-path"]
        assert m.rtt_stats is not None

    def test_path_metrics_throughput_not_none(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        m = result.metrics["test-path"]
        assert m.throughput is not None

    def test_path_identity_preserved(self, dual_server_path) -> None:
        """The Path object stored in PathMetrics must be the same one we passed in."""
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        m = result.metrics["test-path"]
        assert m.path == dual_server_path


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Single path happy-path measurement values
# ══════════════════════════════════════════════════════════════════════════════


class TestSinglePathMeasurements:
    """
    When a healthy server is reachable, measurements must have correct values.
    """

    def test_probe_stats_success_rate_is_100(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=3,
                transfer_bytes=4096,
            )
        )
        ps = result.metrics["test-path"].probe_stats
        assert ps.loss_rate == 0.0
        assert ps.successful == 3

    def test_rtt_stats_mean_is_positive(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=3,
                transfer_bytes=4096,
            )
        )
        rs = result.metrics["test-path"].rtt_stats
        assert rs.mean_ms is not None
        assert rs.mean_ms > 0

    def test_throughput_status_is_success(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        t = result.metrics["test-path"].throughput
        assert t.status == ProbeStatus.SUCCESS

    def test_throughput_mbps_is_positive(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        t = result.metrics["test-path"].throughput
        assert t.throughput_mbps is not None
        assert t.throughput_mbps > 0

    def test_probe_stats_host_port_match_path(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        ps = result.metrics["test-path"].probe_stats
        assert ps.host == dual_server_path.host
        assert ps.port == dual_server_path.port


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Multiple paths
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiplePaths:
    """Multiple paths can be monitored; results stay associated with the right path."""

    def test_two_paths_both_in_result(self, two_dual_servers) -> None:
        (host_a, port_a), (host_b, port_b) = two_dual_servers
        paths = [
            Path(name="alpha", host=host_a, port=port_a),
            Path(name="beta",  host=host_b, port=port_b),
        ]
        result = run(monitor_paths(paths, probe_count=1, transfer_bytes=4096))
        assert "alpha" in result.metrics
        assert "beta" in result.metrics

    def test_two_paths_correct_count(self, two_dual_servers) -> None:
        (host_a, port_a), (host_b, port_b) = two_dual_servers
        paths = [
            Path(name="alpha", host=host_a, port=port_a),
            Path(name="beta",  host=host_b, port=port_b),
        ]
        result = run(monitor_paths(paths, probe_count=1, transfer_bytes=4096))
        assert len(result.metrics) == 2

    def test_two_paths_identity_preserved(self, two_dual_servers) -> None:
        """
        Each path's metrics must carry the correct Path object, not the other
        path's configuration.
        """
        (host_a, port_a), (host_b, port_b) = two_dual_servers
        path_a = Path(name="alpha", host=host_a, port=port_a)
        path_b = Path(name="beta",  host=host_b, port=port_b)
        result = run(monitor_paths([path_a, path_b], probe_count=1, transfer_bytes=4096))

        assert result.metrics["alpha"].path == path_a
        assert result.metrics["beta"].path == path_b

    def test_two_paths_metrics_differ_by_port(self, two_dual_servers) -> None:
        """
        probe_stats.port must match the respective path's port, not be swapped.
        """
        (host_a, port_a), (host_b, port_b) = two_dual_servers
        paths = [
            Path(name="alpha", host=host_a, port=port_a),
            Path(name="beta",  host=host_b, port=port_b),
        ]
        result = run(monitor_paths(paths, probe_count=1, transfer_bytes=4096))

        assert result.metrics["alpha"].probe_stats.port == port_a
        assert result.metrics["beta"].probe_stats.port == port_b

    def test_two_paths_both_succeed(self, two_dual_servers) -> None:
        (host_a, port_a), (host_b, port_b) = two_dual_servers
        paths = [
            Path(name="alpha", host=host_a, port=port_a),
            Path(name="beta",  host=host_b, port=port_b),
        ]
        result = run(monitor_paths(paths, probe_count=2, transfer_bytes=4096))

        assert result.metrics["alpha"].probe_stats.loss_rate == 0.0
        assert result.metrics["beta"].probe_stats.loss_rate == 0.0

    def test_three_paths_all_present(self) -> None:
        """Three independently-served paths all appear in the result."""
        stops = [threading.Event() for _ in range(3)]
        socks_hosts_ports = [_make_dual_server(s) for s in stops]
        paths = [
            Path(name=f"path-{i}", host=h, port=p)
            for i, (_, h, p) in enumerate(socks_hosts_ports)
        ]
        try:
            result = run(monitor_paths(paths, probe_count=1, transfer_bytes=4096))
            assert len(result.metrics) == 3
            for p in paths:
                assert p.name in result.metrics
        finally:
            for stop, (sock, _, _) in zip(stops, socks_hosts_ports):
                stop.set()
                sock.close()


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Concurrency — deterministic synchronisation test
# ══════════════════════════════════════════════════════════════════════════════


class TestConcurrentMonitoring:
    """
    Verify that multiple paths are monitored concurrently, not sequentially.

    Why a Barrier, not elapsed-time?
    ─────────────────────────────────
    Elapsed-time assertions are inherently flaky: a slow CI machine, a
    garbage-collection pause, or OS scheduling jitter can make even genuinely
    concurrent code appear sequential. A threading.Barrier is deterministic:

      barrier = threading.Barrier(N, timeout=T)

    Each of N threads calls barrier.wait(). The barrier releases ALL threads
    only when ALL N have arrived. If monitoring is concurrent, all N threads
    hit the barrier within the timeout window and everyone proceeds. If
    monitoring were sequential, only one thread would ever be running — the
    barrier would wait for the others and time out after T seconds, raising
    threading.BrokenBarrierError.

    We inject the barrier via a custom server fixture whose handler calls
    barrier.wait() before responding to each connection. This forces the
    server handler threads to synchronize, proving that multiple probe threads
    were alive simultaneously.
    """

    def test_two_paths_monitored_concurrently(self) -> None:
        """
        Two paths must be probed at the same time (concurrently), not one after
        the other. We verify this with a threading.Barrier(2).

        The barrier requires exactly 2 threads to arrive before any are released.
        If the paths were monitored sequentially, the second path's probe would
        never start while the first path's probe is still blocking on the barrier
        — the barrier.wait() call on the first path would time out.

        The barrier timeout is set to 5 seconds — generous enough for a
        slow machine, strict enough to catch sequential execution quickly.
        """
        N_PATHS = 2
        BARRIER_TIMEOUT = 5.0  # seconds

        # Barrier is filled by N_PATHS concurrent probe threads.
        barrier = threading.Barrier(N_PATHS, timeout=BARRIER_TIMEOUT)

        stops = [threading.Event() for _ in range(N_PATHS)]
        server_socks = []
        paths = []

        for i, stop in enumerate(stops):
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind(("127.0.0.1", 0))
            server_sock.listen(32)
            server_sock.settimeout(0.05)
            host, port = server_sock.getsockname()
            server_socks.append(server_sock)
            paths.append(Path(name=f"path-{i}", host=host, port=port))

            def _handle(conn: socket.socket, b: threading.Barrier = barrier) -> None:
                try:
                    with conn:
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        reader = conn.makefile("rb")
                        first_line = reader.readline()
                        if first_line == b"PING\n":
                            # Wait for N_PATHS threads to reach this point
                            # simultaneously, proving concurrent monitoring.
                            try:
                                b.wait()  # raises BrokenBarrierError on timeout
                            except threading.BrokenBarrierError:
                                pass  # recorded by barrier state
                            conn.sendall(b"PONG\n")
                        elif first_line.startswith(b"SIZE:"):
                            n_bytes = int(first_line.split(b":")[1].strip())
                            remaining = n_bytes
                            while remaining > 0:
                                to_read = min(65536, remaining)
                                chunk = reader.read(to_read)
                                if not chunk:
                                    break
                                remaining -= len(chunk)
                            conn.sendall(b"DONE\n")
                except OSError:
                    pass

            def _accept_loop(
                sock=server_sock, stop=stop, handle=_handle
            ) -> None:
                while not stop.is_set():
                    try:
                        conn, _ = sock.accept()
                        threading.Thread(
                            target=handle, args=(conn,), daemon=True
                        ).start()
                    except socket.timeout:
                        continue
                    except OSError:
                        break

            threading.Thread(target=_accept_loop, daemon=True).start()

        try:
            result = run(
                monitor_paths(
                    paths,
                    probe_count=1,
                    probe_timeout=BARRIER_TIMEOUT + 1.0,  # generous
                    transfer_bytes=4096,
                )
            )

            # If we get here, the barrier was satisfied — both paths were probed
            # concurrently. Verify the barrier was not broken (timed out).
            assert not barrier.broken, (
                "Barrier was broken: the two paths were NOT probed concurrently. "
                "monitoring appears to be sequential."
            )

            # Sanity: both paths have results
            assert "path-0" in result.metrics
            assert "path-1" in result.metrics

        finally:
            for stop, sock in zip(stops, server_socks):
                stop.set()
                sock.close()

    def test_many_paths_concurrent_returns_all_results(self) -> None:
        """
        4 paths monitored concurrently all return results without one blocking
        another. We do not use a barrier here — we simply assert all 4 paths
        have results and took less time than 4 × probe_timeout sequentially.

        This is a supplementary timing test, kept tolerant (4× margin) so it
        is robust on slow machines.
        """
        N = 4
        PROBE_TIMEOUT = 2.0
        SEQUENTIAL_UPPER_BOUND = N * PROBE_TIMEOUT  # if sequential, takes ~N×T

        stops = [threading.Event() for _ in range(N)]
        socks_hosts_ports = [_make_dual_server(s) for s in stops]
        paths = [
            Path(name=f"p{i}", host=h, port=p)
            for i, (_, h, p) in enumerate(socks_hosts_ports)
        ]

        try:
            t0 = time.perf_counter()
            result = run(
                monitor_paths(
                    paths,
                    probe_count=1,
                    probe_timeout=PROBE_TIMEOUT,
                    transfer_bytes=4096,
                    throughput_timeout=PROBE_TIMEOUT,
                )
            )
            elapsed = time.perf_counter() - t0

            # All paths present
            assert len(result.metrics) == N
            for p in paths:
                assert p.name in result.metrics

            # Concurrent monitoring must complete in less time than sequential
            # would require. We allow a generous 50% margin over sequential to
            # avoid false failures on loaded CI machines.
            # NOTE: This is a supplementary check. The barrier test above is
            # the authoritative concurrency proof.
            assert elapsed < SEQUENTIAL_UPPER_BOUND, (
                f"Monitoring {N} paths took {elapsed:.2f}s, "
                f"which exceeds the sequential upper bound of {SEQUENTIAL_UPPER_BOUND:.2f}s. "
                "This suggests monitoring may be sequential."
            )
        finally:
            for stop, (sock, _, _) in zip(stops, socks_hosts_ports):
                stop.set()
                sock.close()


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Failed path isolation
# ══════════════════════════════════════════════════════════════════════════════


class TestFailedPathIsolation:
    """
    A failed path must not crash, delay, or corrupt the results of healthy paths.
    """

    def test_one_healthy_one_failed_both_in_result(self, dual_server) -> None:
        """
        One healthy path (live server) + one failed path (no server).
        Both must appear in the result, and the healthy path must succeed.
        """
        host, port = dual_server
        healthy_path = Path(name="healthy", host=host, port=port)

        # Get a port with no server (bind, get port, close → nothing listening)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, dead_port = s.getsockname()
        failed_path = Path(name="failed", host="127.0.0.1", port=dead_port)

        result = run(
            monitor_paths(
                [healthy_path, failed_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        assert "healthy" in result.metrics
        assert "failed" in result.metrics

    def test_healthy_path_succeeds_when_sibling_fails(self, dual_server) -> None:
        """
        The healthy path must have 0% probe loss even when a sibling path fails.
        """
        host, port = dual_server
        healthy_path = Path(name="healthy", host=host, port=port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, dead_port = s.getsockname()
        failed_path = Path(name="failed", host="127.0.0.1", port=dead_port)

        result = run(
            monitor_paths(
                [healthy_path, failed_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        healthy_m = result.metrics["healthy"]
        assert healthy_m.probe_stats is not None
        assert healthy_m.probe_stats.loss_rate == 0.0

    def test_failed_path_has_non_success_probe_status(self, dual_server) -> None:
        """
        The failed path must report a non-SUCCESS probe status or 100% loss.
        """
        host, port = dual_server
        healthy_path = Path(name="healthy", host=host, port=port)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, dead_port = s.getsockname()
        failed_path = Path(name="failed", host="127.0.0.1", port=dead_port)

        result = run(
            monitor_paths(
                [healthy_path, failed_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        failed_m = result.metrics["failed"]
        assert failed_m.probe_stats is not None
        # All probes must have failed (100% loss rate)
        assert failed_m.probe_stats.loss_rate == pytest.approx(100.0, abs=0.01)

    def test_unreachable_host_is_isolated(self, dual_server) -> None:
        """
        192.0.2.x (RFC 5737 TEST-NET-1) is documentation-only and unreachable.
        Probes to it time out. The healthy sibling path must still succeed.
        """
        host, port = dual_server
        healthy_path = Path(name="healthy", host=host, port=port)
        # Very short timeout so the test is fast
        unreachable_path = Path(name="unreachable", host="192.0.2.1", port=9999)

        result = run(
            monitor_paths(
                [healthy_path, unreachable_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        assert result.metrics["healthy"].probe_stats.loss_rate == 0.0
        assert result.metrics["unreachable"].probe_stats.loss_rate == pytest.approx(
            100.0, abs=0.01
        )

    def test_failed_path_no_error_field(self, dual_server) -> None:
        """
        A path that fails due to network conditions (connection refused/timeout)
        must NOT have a PathMetrics.error. That field is for unexpected exceptions,
        not for expected network failures.

        run_probes() and measure_throughput() are designed to catch all network
        errors and return result objects with non-SUCCESS status. They should
        not raise, so PathMetrics.error should remain None.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, dead_port = s.getsockname()
        failed_path = Path(name="failed", host="127.0.0.1", port=dead_port)

        result = run(
            monitor_paths(
                [failed_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        failed_m = result.metrics["failed"]
        # PathMetrics.error should be None because run_probes() caught the
        # network failure and returned a ProbeStats with loss_rate=100.
        assert failed_m.error is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Timeout / failure represented in result
# ══════════════════════════════════════════════════════════════════════════════


class TestFailureRepresentation:
    """Failure is captured in result objects, not exceptions."""

    def test_throughput_status_refused_or_timeout_on_dead_port(self) -> None:
        """
        measure_throughput() to a closed port must set status to REFUSED or
        TIMEOUT — never SUCCESS — and throughput_mbps must be None.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, dead_port = s.getsockname()
        failed_path = Path(name="failed", host="127.0.0.1", port=dead_port)

        result = run(
            monitor_paths(
                [failed_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        t = result.metrics["failed"].throughput
        assert t is not None
        assert t.status != ProbeStatus.SUCCESS
        assert t.throughput_mbps is None

    def test_rtt_stats_on_total_probe_failure(self) -> None:
        """
        When all probes fail, rtt_values_ms is empty. compute_rtt_stats([])
        returns count=0 with all None fields. We verify rtt_stats has count==0.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, dead_port = s.getsockname()
        failed_path = Path(name="failed", host="127.0.0.1", port=dead_port)

        result = run(
            monitor_paths(
                [failed_path],
                probe_count=1,
                probe_timeout=0.3,
                transfer_bytes=4096,
                throughput_timeout=0.3,
            )
        )

        rs = result.metrics["failed"].rtt_stats
        assert rs is not None
        assert rs.count == 0
        assert rs.mean_ms is None
        assert rs.jitter_ms is None

    def test_monitor_never_raises_on_all_dead_paths(self) -> None:
        """
        monitor_paths() must always return a MonitorResult — never raise — even
        when every path is unreachable.
        """
        paths = [
            Path(name="dead-1", host="192.0.2.1", port=9001),
            Path(name="dead-2", host="192.0.2.2", port=9002),
        ]
        result = run(
            monitor_paths(
                paths,
                probe_count=1,
                probe_timeout=0.2,
                transfer_bytes=4096,
                throughput_timeout=0.2,
            )
        )
        assert isinstance(result, MonitorResult)
        assert "dead-1" in result.metrics
        assert "dead-2" in result.metrics


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: MonitorResult __str__ smoke test
# ══════════════════════════════════════════════════════════════════════════════


class TestMonitorResultStr:
    """Smoke-test __str__ methods — they must not raise."""

    def test_monitor_result_str_empty(self) -> None:
        result = run(monitor_paths([]))
        s = str(result)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_monitor_result_str_with_data(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        s = str(result)
        assert isinstance(s, str)
        assert "test-path" in s

    def test_path_metrics_str_does_not_raise(self, dual_server_path) -> None:
        result = run(
            monitor_paths(
                [dual_server_path],
                probe_count=1,
                transfer_bytes=4096,
            )
        )
        s = str(result.metrics["test-path"])
        assert isinstance(s, str)
