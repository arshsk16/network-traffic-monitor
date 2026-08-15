"""
core/probe.py — TCP probes: connection probe and RTT measurement.

This module contains two independent probe functions:

  tcp_probe()     — Step 2: attempt a TCP connection, measure handshake time.
  tcp_rtt_probe() — Step 3: establish a TCP connection, send a small probe
                    message, receive a response, measure the round-trip time.

Both functions are deliberately independent of FastAPI, HTTP, and the app/
layer. They know only about sockets and bytes.

Concepts used here:
  - socket.AF_INET       : IPv4 address family
  - socket.SOCK_STREAM   : TCP (reliable, ordered byte stream)
  - socket.settimeout()  : raises socket.timeout if an operation takes too long
  - socket.connect()     : triggers the TCP three-way handshake (SYN/SYN-ACK/ACK)
  - socket.makefile()    : wraps a socket in a file-like interface for readline()
  - time.perf_counter()  : OS high-resolution timer for sub-millisecond timing

Why two separate functions?
  tcp_probe() measures the cost of *establishing* a connection (handshake).
  tcp_rtt_probe() measures the cost of a *request/response cycle* on an
  already-established connection. These are different things. Combining them
  into one function would mix two concepts and make each harder to explain.
"""

import socket
import time
from dataclasses import dataclass
from enum import Enum


class ProbeStatus(str, Enum):
    """
    The three meaningful outcomes of a TCP connection attempt.

    Using an Enum (not raw strings) means:
      - Callers can do `if result.status == ProbeStatus.SUCCESS`
        instead of `if result.status == "success"` (typo-prone)
      - The set of valid statuses is explicit and documented here
    """

    SUCCESS = "success"   # SYN-ACK received; three-way handshake completed
    REFUSED = "refused"   # Server sent RST; nothing is listening on that port
    TIMEOUT = "timeout"   # No response within the deadline; path may be slow or down
    ERROR = "error"       # Unexpected OS-level error (DNS failure, network unreachable, etc.)


@dataclass(frozen=True)
class ProbeResult:
    """
    Immutable result of a single TCP probe attempt.

    Attributes
    ----------
    host:
        The target hostname or IP address that was probed.
    port:
        The target port that was probed.
    status:
        One of ProbeStatus — what happened when we tried to connect.
    elapsed_ms:
        How long the connection attempt took, in milliseconds.
        For SUCCESS, this is the round-trip time of the TCP handshake.
        For REFUSED, this is near-zero (RST arrives immediately).
        For TIMEOUT, this equals the configured timeout.
        For ERROR, this is the time until the error was raised.
    error_message:
        Human-readable description of the error, or None on SUCCESS.

    Why `frozen=True`?
        Results are facts about what happened in the past. They should
        not be mutated after creation. Immutability prevents accidental
        modification and makes results safe to pass around concurrently.
    """

    host: str
    port: int
    status: ProbeStatus
    elapsed_ms: float
    error_message: str | None = None

    def __str__(self) -> str:
        if self.error_message:
            return (
                f"ProbeResult({self.host}:{self.port} -> {self.status.value}, "
                f"{self.elapsed_ms:.2f}ms, error={self.error_message!r})"
            )
        return (
            f"ProbeResult({self.host}:{self.port} -> {self.status.value}, "
            f"{self.elapsed_ms:.2f}ms)"
        )


def tcp_probe(host: str, port: int, timeout: float = 2.0) -> ProbeResult:
    """
    Attempt a TCP connection to `host`:`port` and return a ProbeResult.

    The connection is opened and immediately closed — we only care about
    whether the handshake succeeds and how long it takes. No data is sent.

    Parameters
    ----------
    host:
        Hostname (e.g. "google.com") or IP address (e.g. "8.8.8.8").
    port:
        TCP port to connect to (1-65535).
    timeout:
        Maximum seconds to wait for the handshake to complete.
        If this is exceeded, the status is TIMEOUT.

    Returns
    -------
    ProbeResult
        Always returns a result — never raises. Callers can rely on this.

    Implementation notes
    --------------------
    We use a context manager (`with socket.socket(...) as s`) to guarantee
    the socket is closed even if an exception is raised mid-way. This
    prevents file-descriptor leaks.

    The sequence inside the try block is:
      1. `s.settimeout(timeout)` — arm the timeout before connecting
      2. `start = time.perf_counter()` — start the clock
      3. `s.connect((host, port))` — blocks until handshake or error
      4. `elapsed_ms = (time.perf_counter() - start) * 1000` — stop the clock
      5. Return SUCCESS result

    Multiplication by 1000 converts seconds -> milliseconds.
    Milliseconds are the conventional unit for network latency.
    """
    start = time.perf_counter()  # arm before the try so REFUSED timing is accurate

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ProbeResult(
                host=host,
                port=port,
                status=ProbeStatus.SUCCESS,
                elapsed_ms=elapsed_ms,
            )

    except TimeoutError:
        # socket.timeout is an alias of TimeoutError in Python 3.11+
        # The timeout we set was exceeded — no SYN-ACK arrived in time.
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ProbeResult(
            host=host,
            port=port,
            status=ProbeStatus.TIMEOUT,
            elapsed_ms=elapsed_ms,
            error_message=f"Connection timed out after {timeout}s",
        )

    except ConnectionRefusedError:
        # The OS received an RST (reset) packet — the port is closed.
        # This is fast and immediate; no retry needed.
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ProbeResult(
            host=host,
            port=port,
            status=ProbeStatus.REFUSED,
            elapsed_ms=elapsed_ms,
            error_message="Connection refused (port closed or no service listening)",
        )

    except OSError as exc:
        # Catch-all for other OS-level socket errors:
        # - socket.gaierror (DNS resolution failure, subclass of OSError)
        # - Network unreachable
        # - Host unreachable
        # We don't map these to REFUSED or TIMEOUT — they're genuine errors.
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ProbeResult(
            host=host,
            port=port,
            status=ProbeStatus.ERROR,
            elapsed_ms=elapsed_ms,
            error_message=str(exc),
        )


# ── Step 3: RTT measurement ───────────────────────────────────────────────────

# The probe message and expected response.
# These are arbitrary byte strings — what matters is that the server echoes
# back exactly _EXPECTED_RESPONSE so our measurement is unambiguous.
_PROBE_MESSAGE: bytes = b"PING\n"
_EXPECTED_RESPONSE: bytes = b"PONG\n"


@dataclass(frozen=True)
class RttResult:
    """
    Immutable result of a single application-level RTT measurement.

    This is distinct from ProbeResult because it measures something different:
      ProbeResult.elapsed_ms  -- time for the TCP handshake to complete
      RttResult.rtt_ms        -- time for a message to travel to the server
                                 and a response to travel back

    Attributes
    ----------
    host:
        Target hostname or IP address.
    port:
        Target port.
    status:
        One of ProbeStatus — what happened during the RTT measurement.
    rtt_ms:
        Application-level round-trip time in milliseconds, measured from
        just before socket.sendall() to just after the full response is read.
        None if the measurement failed (timeout, refused, error).
    error_message:
        Human-readable error description, or None on SUCCESS.
    """

    host: str
    port: int
    status: ProbeStatus
    rtt_ms: float | None
    error_message: str | None = None

    def __str__(self) -> str:
        if self.rtt_ms is not None:
            return (
                f"RttResult({self.host}:{self.port} -> {self.status.value}, "
                f"rtt={self.rtt_ms:.2f}ms)"
            )
        return (
            f"RttResult({self.host}:{self.port} -> {self.status.value}, "
            f"error={self.error_message!r})"
        )


def tcp_rtt_probe(host: str, port: int, timeout: float = 2.0) -> RttResult:
    """
    Measure application-level Round-Trip Time (RTT) to a TCP echo server.

    Sequence
    --------
    1. Open a TCP connection to host:port  [handshake NOT timed]
    2. Disable Nagle's algorithm (TCP_NODELAY)
    3. Record start = time.perf_counter()
    4. Send b"PING\\n"
    5. Read until b"PONG\\n" is received
    6. rtt_ms = (time.perf_counter() - start) * 1000

    Why is this RTT and not just connect time?
      The clock starts AFTER the TCP connection is established. We measure
      only the round-trip of our application-level message -- how long it
      takes our bytes to reach the server and the server's bytes to come back.
      The handshake cost is excluded.

    Why TCP_NODELAY?
      By default, TCP uses Nagle's algorithm: it buffers small outgoing
      messages and waits for an ACK before sending, to reduce the number
      of tiny packets on the network. For our 5-byte b"PING\\n" probe this
      would add an artificial delay. TCP_NODELAY disables this buffering
      so the message is sent immediately.

    Why makefile().readline() instead of recv()?
      socket.recv(N) may return fewer than N bytes if the TCP segment
      arrives in pieces. readline() reads until it finds "\\n", correctly
      assembling partial reads automatically.

    Parameters
    ----------
    host:
        Hostname or IP address of the echo server.
    port:
        TCP port of the echo server.
    timeout:
        Seconds to wait for any individual socket operation (connect,
        send, readline). If exceeded, status is TIMEOUT.

    Returns
    -------
    RttResult
        Always returns -- never raises. rtt_ms is None on failure.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)

            # Step 1: Establish TCP connection. NOT timed -- this is handshake
            # cost, measured separately by tcp_probe() in Step 2.
            s.connect((host, port))

            # Step 2: Disable Nagle's algorithm so our probe is sent immediately
            # rather than buffered waiting for more data to batch with.
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Wrap socket for safe line-based reading.
            reader = s.makefile("rb")

            # Steps 3 & 4: Clock starts, probe sent.
            start = time.perf_counter()
            s.sendall(_PROBE_MESSAGE)

            # Step 5: Block until full response line received.
            response = reader.readline()

            # Step 6: Clock stops.
            rtt_ms = (time.perf_counter() - start) * 1000

            # Validate the response -- an unexpected reply means the server
            # does not speak our protocol, not a network problem.
            if response != _EXPECTED_RESPONSE:
                return RttResult(
                    host=host,
                    port=port,
                    status=ProbeStatus.ERROR,
                    rtt_ms=None,
                    error_message=(
                        f"Unexpected response: {response!r} "
                        f"(expected {_EXPECTED_RESPONSE!r})"
                    ),
                )

            return RttResult(
                host=host,
                port=port,
                status=ProbeStatus.SUCCESS,
                rtt_ms=rtt_ms,
            )

    except TimeoutError:
        return RttResult(
            host=host,
            port=port,
            status=ProbeStatus.TIMEOUT,
            rtt_ms=None,
            error_message=f"RTT probe timed out after {timeout}s",
        )

    except ConnectionRefusedError:
        return RttResult(
            host=host,
            port=port,
            status=ProbeStatus.REFUSED,
            rtt_ms=None,
            error_message="Connection refused (port closed or no service listening)",
        )

    except OSError as exc:
        return RttResult(
            host=host,
            port=port,
            status=ProbeStatus.ERROR,
            rtt_ms=None,
            error_message=str(exc),
        )
