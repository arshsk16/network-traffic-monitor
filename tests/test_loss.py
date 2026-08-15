"""
Tests for core/loss.py — probe loss rate measurement.

Test strategy
─────────────
We need to test scenarios where:
  - All probes succeed (0% loss)
  - Some probes fail (partial loss)
  - All probes fail (100% loss)

The challenge: how do we make some probes succeed and some fail in a
deterministic, controlled way — without depending on real network conditions?

Solution: a Selective Echo Server fixture.
  We build a fixture that accepts a configurable number of connections,
  responds with PONG to the first `respond_to` connections, then stops
  accepting. Subsequent connections time out (or are refused after the
  server socket closes). This gives us precise control over how many
  probes succeed.

Why not mock tcp_rtt_probe()?
  Mocking would test that we called a mock correctly, not that our loss
  calculation works with real probe outcomes. The selective server exercises
  actual socket code and gives us real RttResult objects to compute over.

Timing note
───────────
Tests that involve timeouts use a short timeout (0.3s) to keep the suite fast.
We don't assert exact RTT values — only that they are positive and reasonable.
"""

import socket
import threading

import pytest

from core.loss import ProbeStats, run_probes
from core.probe import ProbeStatus

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def echo_server():
    """
    A full echo server that responds to every PING with PONG.
    Used for 0% loss tests.
    """
    stop_event = threading.Event()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(10)
    server_sock.settimeout(0.1)
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
    yield host, port
    stop_event.set()
    server_sock.close()


@pytest.fixture()
def selective_echo_server():
    """
    An echo server that responds to the first N connections with PONG,
    then stops accepting so subsequent probes time out (or are refused).

    Usage
    -----
        host, port, set_limit = selective_echo_server
        set_limit(3)   # respond to first 3 probes only
        stats = run_probes(host, port, count=5, timeout=0.3)
        # → 3 success, 2 failed, 40% loss

    How it simulates failed probes
    ────────────────────────────────
    After responding `limit` times the accept loop exits. The server socket
    is still bound (not closed yet) so the OS accumulates incoming SYNs in
    its backlog queue (listen(10) allows up to 10 pending). Once the backlog
    fills, new connect() calls block until the probe's timeout fires.

    This is a clean, deterministic way to inject failures without relying
    on real network conditions or monkey-patching.

    Yields
    ------
    tuple[str, int, callable]
        (host, port, set_limit) — call set_limit(n) before run_probes().
    """
    stop_event = threading.Event()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(10)
    server_sock.settimeout(0.1)
    host, port = server_sock.getsockname()

    # Shared mutable state: how many responses to give before stopping.
    limit_container: list[int] = [0]

    def set_limit(n: int) -> None:
        limit_container[0] = n

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
        responded = 0
        while not stop_event.is_set():
            if responded >= limit_container[0] > 0:
                # Limit reached: stop accepting. New SYNs queue in the OS
                # backlog and eventually trigger the probe's timeout.
                break
            try:
                conn, _ = server_sock.accept()
                responded += 1
                threading.Thread(target=_handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    threading.Thread(target=_accept_loop, daemon=True).start()
    yield host, port, set_limit
    stop_event.set()
    server_sock.close()


@pytest.fixture()
def port_with_no_server() -> int:
    """Return a port with no server — all probes will fail."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    return port


# ── Tests: return type and structure ─────────────────────────────────────────


class TestProbeStatsStructure:
    """run_probes() must always return a well-formed ProbeStats."""

    def test_returns_probe_stats(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=1)
        assert isinstance(stats, ProbeStats)

    def test_host_and_port_preserved(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=1)
        assert stats.host == host
        assert stats.port == port

    def test_total_equals_count(self, echo_server) -> None:
        """total must always equal the requested probe count."""
        host, port = echo_server
        for count in (1, 3, 5, 10):
            stats = run_probes(host, port, count=count)
            assert stats.total == count

    def test_successful_plus_failed_equals_total(self, echo_server) -> None:
        """successful + failed must always equal total — no probes go missing."""
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        assert stats.successful + stats.failed == stats.total

    def test_raw_results_length_equals_count(self, echo_server) -> None:
        """raw_results must contain exactly one entry per probe."""
        host, port = echo_server
        stats = run_probes(host, port, count=4)
        assert len(stats.raw_results) == 4

    def test_rtt_values_ms_length_equals_successful(self, echo_server) -> None:
        """rtt_values_ms must have one entry per successful probe."""
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        assert len(stats.rtt_values_ms) == stats.successful


# ── Tests: all probes succeed → 0% loss ──────────────────────────────────────


class TestZeroLoss:
    """When every probe succeeds, loss_rate must be 0.0."""

    def test_loss_rate_is_zero(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        assert stats.loss_rate == 0.0

    def test_failed_count_is_zero(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        assert stats.failed == 0

    def test_successful_count_equals_total(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        assert stats.successful == stats.total

    def test_all_rtt_values_positive(self, echo_server) -> None:
        """Every successful probe must record a positive RTT."""
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        assert all(rtt > 0 for rtt in stats.rtt_values_ms)

    def test_rtt_values_count_matches_probe_count(self, echo_server) -> None:
        """On 0% loss, we should have one RTT value per probe."""
        host, port = echo_server
        stats = run_probes(host, port, count=7)
        assert len(stats.rtt_values_ms) == 7


# ── Tests: partial loss ───────────────────────────────────────────────────────


class TestPartialLoss:
    """
    When some probes fail, loss_rate must reflect exactly the failure fraction.

    We use the selective_echo_server to respond to exactly N out of M probes.
    The rest time out (very short timeout to keep tests fast).
    """

    def test_loss_rate_calculation_3_of_5(self, selective_echo_server) -> None:
        """
        3 success, 2 failed out of 5 probes.
        loss_rate = 2/5 * 100 = 40.0%
        """
        host, port, set_limit = selective_echo_server
        set_limit(3)
        stats = run_probes(host, port, count=5, timeout=0.3)
        assert stats.successful == 3
        assert stats.failed == 2
        assert stats.loss_rate == pytest.approx(40.0, abs=0.01)

    def test_loss_rate_calculation_1_of_4(self, selective_echo_server) -> None:
        """
        1 success, 3 failed out of 4 probes.
        loss_rate = 3/4 * 100 = 75.0%
        """
        host, port, set_limit = selective_echo_server
        set_limit(1)
        stats = run_probes(host, port, count=4, timeout=0.3)
        assert stats.successful == 1
        assert stats.failed == 3
        assert stats.loss_rate == pytest.approx(75.0, abs=0.01)

    def test_rtt_values_only_from_successes(self, selective_echo_server) -> None:
        """rtt_values_ms must contain only values from successful probes."""
        host, port, set_limit = selective_echo_server
        set_limit(2)
        stats = run_probes(host, port, count=4, timeout=0.3)
        # Exactly 2 RTT values — one per successful probe
        assert len(stats.rtt_values_ms) == 2
        assert all(rtt > 0 for rtt in stats.rtt_values_ms)

    def test_total_is_always_probe_count(self, selective_echo_server) -> None:
        """total must be the full requested count regardless of failures."""
        host, port, set_limit = selective_echo_server
        set_limit(2)
        stats = run_probes(host, port, count=5, timeout=0.3)
        assert stats.total == 5


# ── Tests: all probes fail → 100% loss ───────────────────────────────────────


class TestFullLoss:
    """When every probe fails, loss_rate must be 100.0 and rtt_values empty."""

    def test_loss_rate_is_100(self, port_with_no_server: int) -> None:
        result = run_probes("127.0.0.1", port_with_no_server, count=3, timeout=0.3)
        assert result.loss_rate == pytest.approx(100.0, abs=0.01)

    def test_successful_is_zero(self, port_with_no_server: int) -> None:
        result = run_probes("127.0.0.1", port_with_no_server, count=3, timeout=0.3)
        assert result.successful == 0

    def test_failed_equals_total(self, port_with_no_server: int) -> None:
        result = run_probes("127.0.0.1", port_with_no_server, count=3, timeout=0.3)
        assert result.failed == result.total

    def test_rtt_values_is_empty(self, port_with_no_server: int) -> None:
        """No RTT values when all probes fail."""
        result = run_probes("127.0.0.1", port_with_no_server, count=3, timeout=0.3)
        assert len(result.rtt_values_ms) == 0

    def test_all_raw_results_are_not_success(self, port_with_no_server: int) -> None:
        """Every raw result must be a non-SUCCESS status."""
        result = run_probes("127.0.0.1", port_with_no_server, count=3, timeout=0.3)
        for r in result.raw_results:
            assert r.status != ProbeStatus.SUCCESS


# ── Tests: probe count is actually respected ──────────────────────────────────


class TestProbeCountRespected:
    """run_probes() must attempt exactly `count` probes, no more, no less."""

    def test_count_1_runs_exactly_1_probe(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=1)
        assert stats.total == 1
        assert len(stats.raw_results) == 1

    def test_count_10_runs_exactly_10_probes(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=10)
        assert stats.total == 10
        assert len(stats.raw_results) == 10


# ── Tests: RTT values are preserved ──────────────────────────────────────────


class TestRttValuesPreserved:
    """Successful probe RTT values must be stored for later use."""

    def test_rtt_values_are_positive_floats(self, echo_server) -> None:
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        for rtt in stats.rtt_values_ms:
            assert isinstance(rtt, float)
            assert rtt > 0

    def test_rtt_values_reasonable_for_loopback(self, echo_server) -> None:
        """Loopback RTTs should be well under 500ms."""
        host, port = echo_server
        stats = run_probes(host, port, count=5)
        for rtt in stats.rtt_values_ms:
            assert rtt < 500

    def test_rtt_values_tuple_is_immutable(self, echo_server) -> None:
        """rtt_values_ms should be a tuple (immutable), not a list."""
        host, port = echo_server
        stats = run_probes(host, port, count=3)
        assert isinstance(stats.rtt_values_ms, tuple)

    def test_raw_results_tuple_is_immutable(self, echo_server) -> None:
        """raw_results should be a tuple (immutable)."""
        host, port = echo_server
        stats = run_probes(host, port, count=3)
        assert isinstance(stats.raw_results, tuple)


# ── Tests: invalid configuration ─────────────────────────────────────────────


class TestInvalidConfiguration:
    """Invalid inputs should raise ValueError immediately, not silently."""

    def test_count_zero_raises_value_error(self, echo_server) -> None:
        host, port = echo_server
        with pytest.raises(ValueError, match="count must be >= 1"):
            run_probes(host, port, count=0)

    def test_count_negative_raises_value_error(self, echo_server) -> None:
        host, port = echo_server
        with pytest.raises(ValueError, match="count must be >= 1"):
            run_probes(host, port, count=-5)

    def test_count_negative_one_raises_value_error(self, echo_server) -> None:
        host, port = echo_server
        with pytest.raises(ValueError):
            run_probes(host, port, count=-1)


# ── Tests: never raises on errors ────────────────────────────────────────────


class TestNeverRaisesOnNetworkErrors:
    """run_probes() must never raise due to network errors — always returns."""

    def test_returns_stats_on_timeout_target(self) -> None:
        """192.0.2.1 is RFC 5737 test-net — packets silently dropped."""
        stats = run_probes("192.0.2.1", 9999, count=2, timeout=0.3)
        assert isinstance(stats, ProbeStats)
        assert stats.loss_rate == pytest.approx(100.0, abs=0.01)

    def test_returns_stats_on_dns_failure(self) -> None:
        """Invalid hostname must produce ERROR probes, not an exception."""
        stats = run_probes("this.host.does.not.exist.invalid", 80, count=2, timeout=1.0)
        assert isinstance(stats, ProbeStats)
        assert stats.successful == 0

    def test_returns_stats_on_unavailable_port(self, port_with_no_server: int) -> None:
        stats = run_probes("127.0.0.1", port_with_no_server, count=2, timeout=0.3)
        assert isinstance(stats, ProbeStats)
