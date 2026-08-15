"""
Tests for core/throughput.py — TCP throughput measurement.

Test strategy
─────────────
Two categories of tests:

1. Mathematical correctness (calculate_throughput_mbps)
   ─────────────────────────────────────────────────────
   We test the pure arithmetic function with controlled inputs.
   No network, no timing variance. All assertions are exact.

   Example: 10 MB in 1.0 second
     bps  = 10_000_000 / 1.0 = 10_000_000
     mbps = 10_000_000 * 8 / 1_000_000 = 80.0 Mbps

2. End-to-end transfer (measure_throughput + throughput_sink_server fixture)
   ─────────────────────────────────────────────────────────────────────────
   A controlled local TCP sink server reads the SIZE header, reads exactly
   that many bytes, and responds with b"DONE\\n".
   We verify:
     - status is SUCCESS
     - bytes_transferred equals what we requested
     - elapsed_seconds > 0
     - throughput_mbps > 0
     - throughput_mbps is mathematically consistent with bytes/elapsed

   We do NOT assert a specific throughput value (e.g., "> 500 Mbps")
   because loopback performance varies across machines and CI environments.
   The test verifies correctness of the measurement, not a performance target.

Why avoid asserting specific throughput numbers?
  Loopback throughput depends on:
    - CPU speed and load
    - OS socket buffer sizes
    - OS version and scheduler
    - Whether the machine is under load (e.g., CI runner contention)
  A test that asserts "> 500 Mbps" would fail on a slow CI machine and
  pass on a fast dev machine — making it an environment test, not a unit test.
  Instead, we assert the _structure_ and _internal consistency_ of the result.
"""

import socket
import threading

import pytest

from core.throughput import (
    ThroughputResult,
    calculate_throughput_mbps,
    measure_throughput,
)
from core.probe import ProbeStatus

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def throughput_sink_server():
    """
    A TCP sink server that:
      1. Reads a SIZE header: b"SIZE:<n>\\n"
      2. Reads exactly <n> bytes of data (and discards them)
      3. Sends b"DONE\\n"

    This implements the server side of the throughput protocol in core/throughput.py.

    Uses a small read buffer (64 KB) to drain the incoming data iteratively,
    matching how a real server would process a bulk upload.

    Yields
    ------
    tuple[str, int]
        (host, port) for the client to connect to.
    """
    stop_event = threading.Event()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(10)
    server_sock.settimeout(0.1)
    host, port = server_sock.getsockname()

    def _recv_header(conn: socket.socket) -> bytes | None:
        """
        Read bytes from conn until a newline is found, returning the full
        header line including the newline. Returns None on connection close.

        Why raw recv() and not makefile().readline()?
          io.BufferedReader.read(N) on a socket is treated as an interactive
          stream on Windows and may return fewer than N bytes even when more
          are available. This causes the subsequent body-read loop to wait
          forever for bytes it already received into the makefile buffer but
          the buffer won't release because read() returned early.
          Using raw recv() throughout avoids all makefile buffering issues.
        """
        buf = b""
        while True:
            byte = conn.recv(1)
            if not byte:
                return None
            buf += byte
            if buf.endswith(b"\n"):
                return buf

    def _handle(conn: socket.socket) -> None:
        try:
            with conn:
                # Step 1: Read the SIZE header with raw recv(), no makefile.
                header = _recv_header(conn)
                if not header or not header.startswith(b"SIZE:"):
                    return
                total = int(header[5:].strip())

                # Step 2: Drain exactly `total` bytes with raw recv().
                received = 0
                while received < total:
                    chunk = conn.recv(min(65536, total - received))
                    if not chunk:
                        return
                    received += len(chunk)

                # Step 3: Acknowledge completion.
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
    yield host, port
    stop_event.set()
    server_sock.close()


@pytest.fixture()
def bad_sink_server():
    """
    A server that sends the WRONG response (not b"DONE\\n").
    Used to test unexpected-response error handling.
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
                # Same: raw recv() for header, no makefile.
                buf = b""
                while b"\n" not in buf:
                    byte = conn.recv(1)
                    if not byte:
                        return
                    buf += byte
                if buf.startswith(b"SIZE:"):
                    total = int(buf[5:].strip())
                    received = 0
                    while received < total:
                        chunk = conn.recv(min(65536, total - received))
                        if not chunk:
                            break
                        received += len(chunk)
                conn.sendall(b"NOPE\n")   # Wrong signal on purpose
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

    threading.Thread(target=_accept_loop, daemon=True).start()
    yield host, port
    stop_event.set()
    server_sock.close()


@pytest.fixture()
def port_with_no_server() -> int:
    """Return a port with no server — connections will be refused or time out."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    return port


# ── Tests: pure math — calculate_throughput_mbps ─────────────────────────────


class TestCalculateThroughputMbps:
    """
    Tests for the pure arithmetic function.
    No network, no timing variance. All assertions are exact.

    Manual verification:
      10_000_000 bytes / 1.0 s = 10_000_000 bytes/s
      × 8 bits/byte = 80_000_000 bits/s
      / 1_000_000 = 80.0 Mbps
    """

    def test_10mb_in_1s_equals_80mbps(self) -> None:
        result = calculate_throughput_mbps(10_000_000, 1.0)
        assert result == pytest.approx(80.0)

    def test_1mb_in_0_1s_equals_80mbps(self) -> None:
        # 1_000_000 / 0.1 = 10_000_000 bps → 80 Mbps
        result = calculate_throughput_mbps(1_000_000, 0.1)
        assert result == pytest.approx(80.0)

    def test_1_byte_in_1s_equals_8_microbps(self) -> None:
        # 1 byte / 1.0 s = 1 byte/s = 8 bits/s = 0.000008 Mbps
        result = calculate_throughput_mbps(1, 1.0)
        assert result == pytest.approx(0.000008)

    def test_125000_bytes_in_1s_equals_1mbps(self) -> None:
        # 1 Mbps = 1_000_000 bits/s = 125_000 bytes/s
        result = calculate_throughput_mbps(125_000, 1.0)
        assert result == pytest.approx(1.0)

    def test_1gb_in_1s_equals_8gbps(self) -> None:
        result = calculate_throughput_mbps(1_000_000_000, 1.0)
        assert result == pytest.approx(8000.0)

    def test_doubling_bytes_doubles_mbps(self) -> None:
        r1 = calculate_throughput_mbps(1_000_000, 1.0)
        r2 = calculate_throughput_mbps(2_000_000, 1.0)
        assert r2 == pytest.approx(r1 * 2)

    def test_halving_time_doubles_mbps(self) -> None:
        r1 = calculate_throughput_mbps(1_000_000, 1.0)
        r2 = calculate_throughput_mbps(1_000_000, 0.5)
        assert r2 == pytest.approx(r1 * 2)

    def test_result_is_float(self) -> None:
        result = calculate_throughput_mbps(1_000_000, 1.0)
        assert isinstance(result, float)

    def test_zero_elapsed_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="elapsed_seconds must be > 0"):
            calculate_throughput_mbps(1_000_000, 0.0)

    def test_negative_elapsed_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="elapsed_seconds must be > 0"):
            calculate_throughput_mbps(1_000_000, -1.0)

    def test_negative_bytes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_throughput_mbps(-1, 1.0)

    def test_zero_bytes_is_valid(self) -> None:
        """Zero bytes transferred is unusual but not a ValueError."""
        result = calculate_throughput_mbps(0, 1.0)
        assert result == 0.0


# ── Tests: successful end-to-end transfer ────────────────────────────────────


class TestThroughputSuccess:
    """measure_throughput() against a real sink server should succeed."""

    def test_returns_throughput_result(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert isinstance(result, ThroughputResult)

    def test_status_is_success(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert result.status == ProbeStatus.SUCCESS

    def test_bytes_transferred_matches_requested(self, throughput_sink_server) -> None:
        """The result must record exactly the bytes we asked to transfer."""
        host, port = throughput_sink_server
        transfer = 64 * 1024  # 64 KiB
        result = measure_throughput(host, port, transfer_bytes=transfer)
        assert result.bytes_transferred == transfer

    def test_elapsed_seconds_is_positive(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert result.elapsed_seconds is not None
        assert result.elapsed_seconds > 0

    def test_throughput_mbps_is_positive(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert result.throughput_mbps is not None
        assert result.throughput_mbps > 0

    def test_throughput_bps_is_positive(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert result.throughput_bps is not None
        assert result.throughput_bps > 0

    def test_no_error_message_on_success(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert result.error_message is None

    def test_host_and_port_preserved(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        assert result.host == host
        assert result.port == port

    def test_mbps_is_internally_consistent(self, throughput_sink_server) -> None:
        """
        throughput_mbps must equal calculate_throughput_mbps(bytes, elapsed).
        This verifies the formula is applied correctly inside measure_throughput().
        """
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        expected = calculate_throughput_mbps(
            result.bytes_transferred, result.elapsed_seconds
        )
        assert result.throughput_mbps == pytest.approx(expected, rel=1e-6)

    def test_bps_times_8_over_million_equals_mbps(self, throughput_sink_server) -> None:
        """Verify the bytes/sec → Mbps conversion formula in the result."""
        host, port = throughput_sink_server
        result = measure_throughput(host, port, transfer_bytes=64 * 1024)
        expected_mbps = result.throughput_bps * 8 / 1_000_000
        assert result.throughput_mbps == pytest.approx(expected_mbps, rel=1e-9)

    def test_different_transfer_sizes_both_succeed(self, throughput_sink_server) -> None:
        """Both small and medium transfers should complete successfully."""
        host, port = throughput_sink_server
        small = measure_throughput(host, port, transfer_bytes=4 * 1024)
        medium = measure_throughput(host, port, transfer_bytes=256 * 1024)
        assert small.status == ProbeStatus.SUCCESS
        assert medium.status == ProbeStatus.SUCCESS
        assert small.bytes_transferred == 4 * 1024
        assert medium.bytes_transferred == 256 * 1024


# ── Tests: custom chunk size ──────────────────────────────────────────────────


class TestCustomChunkSize:
    """measure_throughput() must respect the chunk_size parameter."""

    def test_small_chunk_size_succeeds(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        result = measure_throughput(
            host, port, transfer_bytes=64 * 1024, chunk_size=1024
        )
        assert result.status == ProbeStatus.SUCCESS
        assert result.bytes_transferred == 64 * 1024

    def test_chunk_size_equal_to_transfer_succeeds(self, throughput_sink_server) -> None:
        """When chunk_size >= transfer_bytes, a single sendall() does the job."""
        host, port = throughput_sink_server
        result = measure_throughput(
            host, port, transfer_bytes=4096, chunk_size=65536
        )
        assert result.status == ProbeStatus.SUCCESS
        assert result.bytes_transferred == 4096


# ── Tests: unexpected server response ────────────────────────────────────────


class TestUnexpectedResponse:
    """measure_throughput() must detect a wrong server response."""

    def test_wrong_response_gives_error_status(self, bad_sink_server) -> None:
        host, port = bad_sink_server
        result = measure_throughput(host, port, transfer_bytes=4 * 1024)
        assert result.status == ProbeStatus.ERROR

    def test_wrong_response_throughput_is_none(self, bad_sink_server) -> None:
        host, port = bad_sink_server
        result = measure_throughput(host, port, transfer_bytes=4 * 1024)
        assert result.throughput_mbps is None
        assert result.throughput_bps is None
        assert result.elapsed_seconds is None

    def test_wrong_response_has_error_message(self, bad_sink_server) -> None:
        host, port = bad_sink_server
        result = measure_throughput(host, port, transfer_bytes=4 * 1024)
        assert result.error_message is not None
        assert "Unexpected" in result.error_message


# ── Tests: connection failure ─────────────────────────────────────────────────


class TestConnectionFailure:
    """measure_throughput() must handle connection failures gracefully."""

    def test_unavailable_port_is_not_success(self, port_with_no_server: int) -> None:
        result = measure_throughput(
            "127.0.0.1", port_with_no_server, transfer_bytes=1024, timeout=0.3
        )
        assert result.status != ProbeStatus.SUCCESS

    def test_unavailable_port_throughput_is_none(self, port_with_no_server: int) -> None:
        result = measure_throughput(
            "127.0.0.1", port_with_no_server, transfer_bytes=1024, timeout=0.3
        )
        assert result.throughput_mbps is None

    def test_unavailable_port_has_error_message(self, port_with_no_server: int) -> None:
        result = measure_throughput(
            "127.0.0.1", port_with_no_server, transfer_bytes=1024, timeout=0.3
        )
        assert result.error_message is not None

    def test_dns_failure_returns_error(self) -> None:
        result = measure_throughput("this.host.does.not.exist.invalid", 9999)
        assert result.status == ProbeStatus.ERROR
        assert result.throughput_mbps is None

    def test_timeout_returns_timeout_status(self) -> None:
        """192.0.2.1 is RFC 5737 test-net — packets silently dropped."""
        result = measure_throughput("192.0.2.1", 9999, transfer_bytes=1024, timeout=0.3)
        assert result.status == ProbeStatus.TIMEOUT
        assert result.throughput_mbps is None


# ── Tests: invalid configuration ─────────────────────────────────────────────


class TestInvalidConfiguration:
    """Invalid inputs must be rejected with ValueError before any network I/O."""

    def test_transfer_bytes_zero_raises(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        with pytest.raises(ValueError, match="transfer_bytes must be >= 1"):
            measure_throughput(host, port, transfer_bytes=0)

    def test_transfer_bytes_negative_raises(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        with pytest.raises(ValueError, match="transfer_bytes must be >= 1"):
            measure_throughput(host, port, transfer_bytes=-1)

    def test_chunk_size_zero_raises(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            measure_throughput(host, port, transfer_bytes=1024, chunk_size=0)

    def test_chunk_size_negative_raises(self, throughput_sink_server) -> None:
        host, port = throughput_sink_server
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            measure_throughput(host, port, transfer_bytes=1024, chunk_size=-1)


# ── Tests: never raises on network errors ────────────────────────────────────


class TestNeverRaises:
    """measure_throughput() must always return a ThroughputResult, never raise."""

    def test_returns_result_on_connection_failure(self, port_with_no_server: int) -> None:
        result = measure_throughput(
            "127.0.0.1", port_with_no_server, transfer_bytes=1024, timeout=0.3
        )
        assert isinstance(result, ThroughputResult)

    def test_returns_result_on_timeout(self) -> None:
        result = measure_throughput("192.0.2.1", 9999, transfer_bytes=1024, timeout=0.3)
        assert isinstance(result, ThroughputResult)

    def test_returns_result_on_dns_failure(self) -> None:
        result = measure_throughput("this.host.does.not.exist.invalid", 9999)
        assert isinstance(result, ThroughputResult)
