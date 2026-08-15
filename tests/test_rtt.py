"""
Tests for tcp_rtt_probe() — application-level RTT measurement.

Test strategy
─────────────
All tests use a controlled local TCP echo server that:
  1. Accepts a connection.
  2. Reads until it receives b"PING\\n".
  3. Immediately sends back b"PONG\\n".
  4. Closes the connection.

Why an echo server rather than a mock?
  A mock would test that we called the mock correctly. A real echo server
  exercises actual socket operations: the OS network stack, kernel buffering,
  the TCP_NODELAY socket option, and the makefile().readline() path. If the
  RTT probe works against this real server, we know the socket code works.

Why is the server in this test file and not a shared fixture in conftest.py?
  The Step 2 local_tcp_server (in test_probe.py) just accepts and drops
  connections — it never reads or writes data. The RTT echo server must
  read "PING\\n" and write "PONG\\n". These are different servers with
  different behaviours. Keeping them separate avoids coupling Step 2 and
  Step 3 tests.

Timing assertions
─────────────────
We test that rtt_ms > 0 and < a generous upper bound (500ms for loopback).
We do NOT test for an exact value because OS scheduling can make even a
loopback RTT vary by several milliseconds from run to run.
"""

import socket
import threading

import pytest

from core.probe import ProbeStatus, RttResult, tcp_rtt_probe

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def echo_server():
    """
    A minimal TCP echo server that responds to our PING/PONG protocol.

    For each accepted connection the server:
      1. Reads until b"PING\\n" is received.
      2. Sends b"PONG\\n" immediately.
      3. Closes the connection.

    The server runs in a daemon background thread so it never blocks pytest.

    Why handle connections in a new thread per connection?
      The server needs to stay available to accept the next connection while
      handling the current one. For our tests a single sequential handler is
      sufficient — each test opens one connection — but threading makes this
      explicit and correct.

    Yields
    ------
    tuple[str, int]
        (host, port) the RTT probe should connect to.
    """
    stop_event = threading.Event()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(5)
    server_sock.settimeout(0.1)

    host, port = server_sock.getsockname()

    def _handle_connection(conn: socket.socket) -> None:
        """Read PING, send PONG, close. Runs in its own thread."""
        try:
            with conn:
                # TCP_NODELAY on the server side too, so PONG is sent immediately.
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                reader = conn.makefile("rb")
                data = reader.readline()  # blocks until "\n" or connection closes
                if data == b"PING\n":
                    conn.sendall(b"PONG\n")
                # If data is anything else, we close without responding.
                # This exercises the "unexpected response" error path.
        except OSError:
            pass  # Connection closed by client during teardown — expected.

    def _accept_loop() -> None:
        """Accept connections and spawn a handler thread for each one."""
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                t = threading.Thread(target=_handle_connection, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()

    yield host, port

    # ── Teardown ──
    stop_event.set()
    server_sock.close()
    thread.join(timeout=1.0)


@pytest.fixture()
def bad_echo_server():
    """
    A TCP server that sends the WRONG response — not b"PONG\\n".

    Used to test that tcp_rtt_probe() correctly detects and reports an
    unexpected response as ProbeStatus.ERROR, not as SUCCESS.

    Yields
    ------
    tuple[str, int]
        (host, port) to connect to.
    """
    stop_event = threading.Event()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(5)
    server_sock.settimeout(0.1)

    host, port = server_sock.getsockname()

    def _handle_bad(conn: socket.socket) -> None:
        try:
            with conn:
                reader = conn.makefile("rb")
                reader.readline()           # read (and discard) whatever the client sent
                conn.sendall(b"NOPE\n")    # wrong response — client should detect this
        except OSError:
            pass

    def _accept_loop() -> None:
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                threading.Thread(target=_handle_bad, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except OSError:
                break

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()

    yield host, port

    stop_event.set()
    server_sock.close()
    thread.join(timeout=1.0)


@pytest.fixture()
def port_with_no_server() -> int:
    """Return a port number that has no server listening on it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    return port


# ── Tests: successful RTT measurement ─────────────────────────────────────────


class TestRttProbeSuccess:
    """tcp_rtt_probe() against a live echo server should return SUCCESS."""

    def test_returns_rtt_result(self, echo_server) -> None:
        """Return type must be RttResult."""
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert isinstance(result, RttResult)

    def test_status_is_success(self, echo_server) -> None:
        """Status must be SUCCESS when the echo server responds correctly."""
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert result.status == ProbeStatus.SUCCESS

    def test_rtt_ms_is_not_none(self, echo_server) -> None:
        """rtt_ms must be populated on success."""
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert result.rtt_ms is not None

    def test_rtt_ms_is_positive(self, echo_server) -> None:
        """
        RTT must always be > 0.
        Even on loopback the OS takes some time to process the message.
        """
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert result.rtt_ms > 0

    def test_rtt_ms_is_reasonable_for_loopback(self, echo_server) -> None:
        """
        Loopback RTT should complete well within 500ms.

        We use 500ms (not 10ms) as the upper bound to avoid flakiness on
        heavily loaded CI machines. The actual value is typically < 5ms.
        """
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert result.rtt_ms < 500

    def test_host_and_port_preserved(self, echo_server) -> None:
        """Result must echo back the host and port that was probed."""
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert result.host == host
        assert result.port == port

    def test_no_error_message_on_success(self, echo_server) -> None:
        """No error message when the RTT probe succeeds."""
        host, port = echo_server
        result = tcp_rtt_probe(host, port)
        assert result.error_message is None

    def test_multiple_probes_all_succeed(self, echo_server) -> None:
        """
        Running the probe several times should consistently succeed.
        This exercises the echo server's ability to handle multiple
        sequential connections, and gives us confidence the result
        is repeatable rather than a lucky one-off.
        """
        host, port = echo_server
        results = [tcp_rtt_probe(host, port) for _ in range(5)]
        for result in results:
            assert result.status == ProbeStatus.SUCCESS
            assert result.rtt_ms is not None
            assert result.rtt_ms > 0


# ── Tests: this is RTT, not connect time ──────────────────────────────────────


class TestRttIsApplicationLevel:
    """
    Conceptual tests that distinguish RTT from TCP connect time.

    We can't directly measure "what the probe excluded" in a unit test,
    but we can verify the structural guarantees that make this RTT:
      - rtt_ms is measured AFTER connect() completes
      - Multiple probes on the same machine are consistent
    """

    def test_rtt_is_consistent_across_samples(self, echo_server) -> None:
        """
        Multiple RTT samples on loopback should be in the same ballpark.

        We check that the max sample is within 100ms of the min sample.
        On loopback this should easily be < 10ms spread; 100ms is a
        generous bound that accounts for OS scheduling noise.

        This consistency check would fail if we were measuring something
        noisy like wall-clock time or thread scheduling overhead.
        """
        host, port = echo_server
        samples = [tcp_rtt_probe(host, port).rtt_ms for _ in range(5)]
        assert all(s is not None for s in samples)
        spread = max(samples) - min(samples)
        assert spread < 100, f"RTT spread too high: {spread:.2f}ms — samples: {samples}"


# ── Tests: timeout ────────────────────────────────────────────────────────────


class TestRttProbeTimeout:
    """tcp_rtt_probe() must respect the timeout and return TIMEOUT status."""

    def test_timeout_on_unreachable_address(self) -> None:
        """
        192.0.2.0/24 is TEST-NET-1 (RFC 5737) — no real host responds.
        The connect() call will block until our short timeout fires.
        """
        result = tcp_rtt_probe("192.0.2.1", 9999, timeout=0.3)
        assert result.status == ProbeStatus.TIMEOUT

    def test_rtt_ms_is_none_on_timeout(self) -> None:
        """rtt_ms must be None when the probe timed out — no RTT was measured."""
        result = tcp_rtt_probe("192.0.2.1", 9999, timeout=0.3)
        assert result.rtt_ms is None

    def test_timeout_has_error_message(self) -> None:
        """Timeout result must include a human-readable explanation."""
        result = tcp_rtt_probe("192.0.2.1", 9999, timeout=0.3)
        assert result.error_message is not None
        assert len(result.error_message) > 0


# ── Tests: connection failure ─────────────────────────────────────────────────


class TestRttProbeConnectionFailure:
    """tcp_rtt_probe() must handle connection failures gracefully."""

    def test_unavailable_port_is_not_success(self, port_with_no_server: int) -> None:
        """A port with no server must not report SUCCESS."""
        result = tcp_rtt_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert result.status != ProbeStatus.SUCCESS

    def test_rtt_ms_is_none_on_failure(self, port_with_no_server: int) -> None:
        """rtt_ms must be None when no connection could be made."""
        result = tcp_rtt_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert result.rtt_ms is None

    def test_invalid_hostname_returns_error(self) -> None:
        """DNS failure must return ERROR, not raise an exception."""
        result = tcp_rtt_probe("this.hostname.does.not.exist.invalid", 80)
        assert result.status == ProbeStatus.ERROR
        assert result.rtt_ms is None
        assert result.error_message is not None


# ── Tests: unexpected server response ────────────────────────────────────────


class TestRttProbeUnexpectedResponse:
    """tcp_rtt_probe() must detect when a server sends the wrong reply."""

    def test_wrong_response_gives_error_status(self, bad_echo_server) -> None:
        """
        If the server sends something other than b"PONG\\n", the status
        must be ERROR — we connected successfully but the protocol was wrong.
        """
        host, port = bad_echo_server
        result = tcp_rtt_probe(host, port)
        assert result.status == ProbeStatus.ERROR

    def test_wrong_response_rtt_ms_is_none(self, bad_echo_server) -> None:
        """rtt_ms must be None when the response was unexpected."""
        host, port = bad_echo_server
        result = tcp_rtt_probe(host, port)
        assert result.rtt_ms is None

    def test_wrong_response_has_error_message(self, bad_echo_server) -> None:
        """Error message must mention the unexpected response."""
        host, port = bad_echo_server
        result = tcp_rtt_probe(host, port)
        assert result.error_message is not None
        assert "Unexpected response" in result.error_message


# ── Tests: never raises ───────────────────────────────────────────────────────


class TestRttProbeNeverRaises:
    """tcp_rtt_probe() must always return an RttResult, never raise."""

    def test_returns_result_on_timeout(self) -> None:
        result = tcp_rtt_probe("192.0.2.1", 9999, timeout=0.3)
        assert isinstance(result, RttResult)

    def test_returns_result_on_dns_failure(self) -> None:
        result = tcp_rtt_probe("this.hostname.does.not.exist.invalid", 80)
        assert isinstance(result, RttResult)

    def test_returns_result_on_connection_failure(self, port_with_no_server: int) -> None:
        result = tcp_rtt_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert isinstance(result, RttResult)
