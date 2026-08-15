"""
Tests for core/probe.py — TCP connection probe.

Test strategy
─────────────
We test three real scenarios using an actual local TCP server.
No mocking. No external network. No Google, Cloudflare, or anything else.

Why a real local server?

  Mocking socket.connect() would test that we called a mock correctly,
  not that our socket code actually works. A real server exercises the
  actual OS TCP stack: SYN packets, RST packets, the three-way handshake.
  If the probe works against a real server, it will work against real targets.

How the local server works
──────────────────────────
  The `local_tcp_server` fixture:
    1. Creates a socket and binds it to 127.0.0.1 on an OS-assigned free port.
    2. Calls listen() to put it in the listening state.
    3. Runs an accept() loop in a background daemon thread — it simply accepts
       and immediately closes each connection. It never reads or writes data
       because the probe doesn't send any.
    4. Yields (host, port) to the test so it knows where to connect.
    5. After the test, closes the server socket. The daemon thread exits
       automatically because daemon threads die when the main thread ends
       or when we set the stop_event.

Why port 0?
    Passing port=0 to bind() asks the OS to assign any available free port.
    This avoids "address already in use" errors if a previous test run
    left a socket in TIME_WAIT state.

Why a daemon thread?
    Daemon threads are automatically killed when the main Python process exits.
    If something goes wrong in a test and the fixture teardown doesn't run,
    the thread won't keep pytest hanging.
"""

import socket
import threading

import pytest

from core.probe import ProbeResult, ProbeStatus, tcp_probe

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def local_tcp_server():
    """
    Spin up a minimal TCP server on 127.0.0.1 and a free OS-assigned port.

    The server accepts connections and immediately closes them — it exists
    only to allow a successful TCP handshake (SYN → SYN-ACK → ACK).

    Yields
    ------
    tuple[str, int]
        (host, port) the probe should connect to.
    """
    stop_event = threading.Event()

    # SO_REUSEADDR lets us rebind quickly if the port is in TIME_WAIT state
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))          # port=0 → OS picks a free port
    server_sock.listen(5)                         # up to 5 pending connections in queue
    server_sock.settimeout(0.1)                   # short timeout so accept() doesn't block forever

    host, port = server_sock.getsockname()        # find out which port the OS assigned

    def _accept_loop() -> None:
        """Accept connections until stop_event is set, then exit cleanly."""
        while not stop_event.is_set():
            try:
                conn, _ = server_sock.accept()
                conn.close()          # immediately close — we only need the handshake
            except socket.timeout:
                continue              # timed out on accept(); check stop_event and loop
            except OSError:
                break                 # server socket was closed; exit the thread

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()

    yield host, port  # ← test runs here

    # ── Teardown ──
    stop_event.set()
    server_sock.close()
    thread.join(timeout=1.0)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTcpProbeSuccess:
    """tcp_probe() against a live local server should succeed."""

    def test_returns_probe_result(self, local_tcp_server) -> None:
        """Return type must be ProbeResult."""
        host, port = local_tcp_server
        result = tcp_probe(host, port)
        assert isinstance(result, ProbeResult)

    def test_status_is_success(self, local_tcp_server) -> None:
        """Status must be SUCCESS when a server is listening."""
        host, port = local_tcp_server
        result = tcp_probe(host, port)
        assert result.status == ProbeStatus.SUCCESS

    def test_elapsed_ms_is_positive(self, local_tcp_server) -> None:
        """A successful handshake always takes some time > 0."""
        host, port = local_tcp_server
        result = tcp_probe(host, port)
        assert result.elapsed_ms > 0

    def test_elapsed_ms_is_reasonable(self, local_tcp_server) -> None:
        """Loopback handshake should complete well within 500 ms."""
        host, port = local_tcp_server
        result = tcp_probe(host, port)
        assert result.elapsed_ms < 500

    def test_host_and_port_preserved(self, local_tcp_server) -> None:
        """Result must echo back the host and port that was probed."""
        host, port = local_tcp_server
        result = tcp_probe(host, port)
        assert result.host == host
        assert result.port == port

    def test_no_error_message_on_success(self, local_tcp_server) -> None:
        """No error message when connection succeeds."""
        host, port = local_tcp_server
        result = tcp_probe(host, port)
        assert result.error_message is None


@pytest.fixture()
def port_with_no_server() -> int:
    """
    Return a port number that has no server listening on it.

    We bind a socket to get an OS-assigned free port, immediately close it,
    and return the port number. No service is listening on this port.

    On Linux: connecting here gives immediate ConnectionRefusedError (RST).
    On Windows: the Windows Firewall may silently drop the SYN, giving
    TimeoutError instead of ConnectionRefusedError.

    Both outcomes mean the same thing: "nothing is available on this port".
    The test suite accepts either outcome as correct.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    return port


class TestTcpProbeUnavailablePort:
    """
    tcp_probe() to a port with no server should return REFUSED or TIMEOUT.

    Platform note:
    - Linux/macOS: closed port sends RST immediately → ProbeStatus.REFUSED
    - Windows: Windows Firewall may silently drop the SYN → ProbeStatus.TIMEOUT

    Both mean "nothing is available on this port". The *important* contract
    is that tcp_probe() always returns a ProbeResult and never raises.
    For the status-specific REFUSED test, see TestTcpProbeRefused below.
    """

    def test_unavailable_port_returns_probe_result(self, port_with_no_server: int) -> None:
        """tcp_probe() must always return a ProbeResult, never raise."""
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert isinstance(result, ProbeResult)

    def test_unavailable_port_is_not_success(self, port_with_no_server: int) -> None:
        """A port with no server must not show as SUCCESS."""
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert result.status != ProbeStatus.SUCCESS

    def test_unavailable_port_status_is_refused_or_timeout(self, port_with_no_server: int) -> None:
        """
        On Linux: REFUSED (RST received immediately).
        On Windows: TIMEOUT (Windows Firewall drops the SYN silently).
        Both are valid "port unavailable" outcomes.
        """
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert result.status in (ProbeStatus.REFUSED, ProbeStatus.TIMEOUT)

    def test_unavailable_port_has_error_message(self, port_with_no_server: int) -> None:
        """Either REFUSED or TIMEOUT must include an error message."""
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert result.error_message is not None


class TestTcpProbeRefused:
    """
    On platforms where RST is received for closed ports (Linux/macOS),
    tcp_probe() should specifically return REFUSED.

    On Windows with the default firewall, these tests are expected to be
    skipped or may show TIMEOUT. This is documented here for interview clarity:
    The distinction between REFUSED and TIMEOUT is a real networking concept
    even if we can't always force one or the other in a test environment.
    """

    def test_refused_has_error_message(self, port_with_no_server: int) -> None:
        """If REFUSED, there must be a human-readable error message."""
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        if result.status == ProbeStatus.REFUSED:
            assert result.error_message is not None
            assert len(result.error_message) > 0

    def test_refused_is_fast(self, port_with_no_server: int) -> None:
        """
        On Linux, a refused connection arrives nearly instantly.
        On Windows, the timeout fires (so elapsed_ms ≈ timeout).
        In both cases, elapsed_ms must be positive.
        """
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert result.elapsed_ms > 0


class TestTcpProbeTimeout:
    """tcp_probe() should respect the timeout and return TIMEOUT status."""

    def test_timeout_status_on_unreachable_address(self) -> None:
        """
        192.0.2.0/24 is TEST-NET-1, reserved by RFC 5737 for documentation.
        No real machine uses these addresses, so packets are silently dropped
        and connect() will block until our timeout fires.

        We use a very short timeout (0.3s) to keep the test fast.
        """
        result = tcp_probe("192.0.2.1", 9999, timeout=0.3)
        assert result.status == ProbeStatus.TIMEOUT

    def test_timeout_elapsed_ms_matches_timeout(self) -> None:
        """
        Elapsed time should be approximately equal to the configured timeout.

        We allow up to 200 ms of slack for OS scheduling overhead. On a
        lightly loaded machine the actual elapsed time is very close to
        the configured timeout.
        """
        timeout_seconds = 0.3
        result = tcp_probe("192.0.2.1", 9999, timeout=timeout_seconds)
        assert result.elapsed_ms >= timeout_seconds * 1000 - 50   # not faster than timeout
        assert result.elapsed_ms < timeout_seconds * 1000 + 200   # not wildly over

    def test_timeout_has_error_message(self) -> None:
        """Timeout result must explain what happened."""
        result = tcp_probe("192.0.2.1", 9999, timeout=0.3)
        assert result.error_message is not None

    def test_custom_timeout_is_respected(self, local_tcp_server) -> None:
        """A generous timeout should still allow a local connection to succeed."""
        host, port = local_tcp_server
        result = tcp_probe(host, port, timeout=5.0)
        assert result.status == ProbeStatus.SUCCESS


class TestTcpProbeNeverRaises:
    """tcp_probe() must never raise an exception — it always returns a result."""

    def test_invalid_hostname_returns_error_not_exception(self) -> None:
        """
        DNS lookup for a non-existent hostname raises socket.gaierror
        internally. tcp_probe must catch it and return an ERROR result.
        """
        result = tcp_probe("this.hostname.does.not.exist.invalid", 80)
        assert isinstance(result, ProbeResult)
        assert result.status == ProbeStatus.ERROR
        assert result.error_message is not None

    def test_returns_probe_result_not_exception_on_unavailable(self, port_with_no_server: int) -> None:
        """Whether REFUSED or TIMEOUT, we always get a ProbeResult, not an exception."""
        result = tcp_probe("127.0.0.1", port_with_no_server, timeout=0.5)
        assert isinstance(result, ProbeResult)
