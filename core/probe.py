"""
core/probe.py — TCP connection probe.

This module contains the single responsibility of attempting a TCP connection
to a target host and port, measuring how long it takes, and returning a
structured result.

Deliberately has zero imports from app/ or FastAPI. The probe knows nothing
about HTTP, routing, or the API layer. It only knows about sockets.

Concepts used here:
  - socket.AF_INET       : IPv4 address family
  - socket.SOCK_STREAM   : TCP (reliable, ordered byte stream)
  - socket.settimeout()  : raises socket.timeout if connect() takes too long
  - socket.connect()     : triggers the TCP three-way handshake (SYN/SYN-ACK/ACK)
  - time.perf_counter()  : OS high-resolution timer for sub-millisecond timing
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
                f"ProbeResult({self.host}:{self.port} → {self.status.value}, "
                f"{self.elapsed_ms:.2f}ms, error={self.error_message!r})"
            )
        return (
            f"ProbeResult({self.host}:{self.port} → {self.status.value}, "
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
        TCP port to connect to (1–65535).
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

    Multiplication by 1000 converts seconds → milliseconds.
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
