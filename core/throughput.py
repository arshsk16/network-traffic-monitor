"""
core/throughput.py — Application-level TCP throughput measurement.

This module measures the rate at which TCP can transfer a configurable
amount of data to a controlled echo server under our test conditions.

Important terminology
─────────────────────
We call our result `measured_tcp_throughput_mbps`. We do NOT claim it
measures a physical link's maximum bandwidth. Here's why:

  Bandwidth:    the theoretical capacity of a link (set by hardware/ISP)
  Throughput:   the actual data transfer rate achieved in a real transfer

  Our measurement is application-level, loopback-local, and affected by:
    - TCP slow start (each new connection ramps up)
    - OS socket send/receive buffer sizes
    - CPU and memory bandwidth (loopback is CPU-bound, not wire-bound)
    - OS process scheduling (preemption adds elapsed time)

  In an interview: "We measure application-level TCP throughput over a
  local echo transfer. This tells us how fast data can move through the
  TCP stack between two endpoints in our environment. It is not the same
  as the WAN link's rated bandwidth."

Transfer protocol
─────────────────
1. Client connects (handshake — NOT timed)
2. Client sends a header: b"SIZE:<n_bytes>\\n"
   Server reads the header and prepares to receive exactly n_bytes.
3. Client sends n_bytes of data in chunks (timer STARTS here)
4. Server reads and discards n_bytes (it does not echo data back —
   that would double the measurement noise)
5. Server sends b"DONE\\n" after reading all n_bytes (timer STOPS here)

Why not echo the data back?
  If the server echoed every byte, we would be measuring upload AND download
  throughput at the same time on the same loopback. That introduces TCP
  receive-window contention and makes the result harder to interpret.
  The one-way upload measurement is cleaner: measures how fast the client
  can push data into the socket and have it consumed.

Timer placement
───────────────
  Timer starts: just before the first data chunk is sent via sendall()
  Timer stops: immediately after readline() returns b"DONE\\n"
  Excluded: connect(), header exchange

  This isolates the data-transfer cost from connection-setup overhead.
  Connection setup (the three-way handshake) is already measured by Step 2.

Throughput formula
──────────────────
  throughput_bps  = bytes_transferred / elapsed_seconds
  throughput_mbps = throughput_bps * 8 / 1_000_000

  The ×8 converts bytes to bits (1 byte = 8 bits).
  The ÷1_000_000 converts bits/s to Mbps (mega = 10^6, SI prefix).
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from core.probe import ProbeStatus

# Default transfer parameters.
# 1 MB is large enough for TCP to exit slow-start and settle to steady-state
# throughput, while being small enough to complete quickly in tests.
DEFAULT_TRANSFER_BYTES: int = 1 * 1024 * 1024   # 1 MiB
DEFAULT_CHUNK_SIZE: int = 64 * 1024              # 64 KiB per sendall() call

# Application protocol tokens.
_DONE_SIGNAL: bytes = b"DONE\n"


@dataclass(frozen=True)
class ThroughputResult:
    """
    Result of a single TCP throughput measurement.

    Attributes
    ----------
    host:
        Target hostname or IP address.
    port:
        Target port.
    status:
        Outcome of the transfer (ProbeStatus.SUCCESS / TIMEOUT / REFUSED / ERROR).
    bytes_transferred:
        Number of application-data bytes the client sent. None on failure.
    elapsed_seconds:
        Wall-clock time from first data chunk sent to DONE received. None on failure.
    throughput_bps:
        Calculated bytes per second. None on failure.
    throughput_mbps:
        Calculated megabits per second. None on failure.
    error_message:
        Human-readable error description, or None on SUCCESS.
    """

    host: str
    port: int
    status: ProbeStatus
    bytes_transferred: int | None
    elapsed_seconds: float | None
    throughput_bps: float | None
    throughput_mbps: float | None
    error_message: str | None = None

    def __str__(self) -> str:
        if self.throughput_mbps is not None:
            return (
                f"ThroughputResult({self.host}:{self.port} -> {self.status.value}, "
                f"{self.bytes_transferred:,} bytes in {self.elapsed_seconds:.3f}s, "
                f"{self.throughput_mbps:.1f} Mbps)"
            )
        return (
            f"ThroughputResult({self.host}:{self.port} -> {self.status.value}, "
            f"error={self.error_message!r})"
        )


def measure_throughput(
    host: str,
    port: int,
    transfer_bytes: int = DEFAULT_TRANSFER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = 10.0,
) -> ThroughputResult:
    """
    Measure TCP throughput by uploading `transfer_bytes` to a sink server.

    Parameters
    ----------
    host:
        Hostname or IP address of the throughput sink server.
    port:
        TCP port of the sink server.
    transfer_bytes:
        Total bytes of application data to send. Must be >= 1.
    chunk_size:
        Bytes per sendall() call. Smaller chunks increase syscall overhead;
        larger chunks reduce it. 64 KiB is a reasonable default.
    timeout:
        Socket-level timeout in seconds for connect and transfer operations.

    Returns
    -------
    ThroughputResult
        Always returns — never raises. On failure, numeric fields are None.

    Raises
    ------
    ValueError
        If transfer_bytes < 1 or chunk_size < 1.
    """
    if transfer_bytes < 1:
        raise ValueError(f"transfer_bytes must be >= 1, got {transfer_bytes!r}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size!r}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)

            # Step 1: Establish connection — NOT timed.
            s.connect((host, port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            reader = s.makefile("rb")

            # Step 2: Send the SIZE header so the server knows when to stop.
            # This is also NOT timed — it is protocol negotiation, not transfer.
            s.sendall(f"SIZE:{transfer_bytes}\n".encode())

            # Step 3: Send data in chunks — timer STARTS before first chunk.
            #
            # We pre-allocate one chunk-sized bytes object and reuse it for
            # every sendall(). This avoids allocating `transfer_bytes` of
            # memory all at once (which would be wasteful for large transfers).
            chunk = bytes(chunk_size)   # zero-filled bytes — content doesn't matter
            remaining = transfer_bytes

            start = time.perf_counter()  # ← timer starts here

            while remaining > 0:
                to_send = min(chunk_size, remaining)
                # Use a slice only if this is the last (smaller) chunk.
                s.sendall(chunk if to_send == chunk_size else chunk[:to_send])
                remaining -= to_send

            # Step 4: Wait for server's DONE confirmation.
            # The timer measures the full one-way transfer including TCP ACKs
            # for the last sent bytes — which is correct: we want to know when
            # the data was actually received, not just when it left our send buffer.
            response = reader.readline()

            elapsed_seconds = time.perf_counter() - start  # ← timer stops here

            if response != _DONE_SIGNAL:
                return ThroughputResult(
                    host=host,
                    port=port,
                    status=ProbeStatus.ERROR,
                    bytes_transferred=None,
                    elapsed_seconds=None,
                    throughput_bps=None,
                    throughput_mbps=None,
                    error_message=(
                        f"Unexpected server response: {response!r} "
                        f"(expected {_DONE_SIGNAL!r})"
                    ),
                )

            # Step 5: Calculate throughput.
            throughput_bps = transfer_bytes / elapsed_seconds
            throughput_mbps = throughput_bps * 8 / 1_000_000

            return ThroughputResult(
                host=host,
                port=port,
                status=ProbeStatus.SUCCESS,
                bytes_transferred=transfer_bytes,
                elapsed_seconds=elapsed_seconds,
                throughput_bps=throughput_bps,
                throughput_mbps=throughput_mbps,
            )

    except TimeoutError:
        return ThroughputResult(
            host=host,
            port=port,
            status=ProbeStatus.TIMEOUT,
            bytes_transferred=None,
            elapsed_seconds=None,
            throughput_bps=None,
            throughput_mbps=None,
            error_message=f"Transfer timed out after {timeout}s",
        )

    except ConnectionRefusedError:
        return ThroughputResult(
            host=host,
            port=port,
            status=ProbeStatus.REFUSED,
            bytes_transferred=None,
            elapsed_seconds=None,
            throughput_bps=None,
            throughput_mbps=None,
            error_message="Connection refused (port closed or no service listening)",
        )

    except OSError as exc:
        return ThroughputResult(
            host=host,
            port=port,
            status=ProbeStatus.ERROR,
            bytes_transferred=None,
            elapsed_seconds=None,
            throughput_bps=None,
            throughput_mbps=None,
            error_message=str(exc),
        )


def calculate_throughput_mbps(bytes_transferred: int, elapsed_seconds: float) -> float:
    """
    Calculate throughput in Mbps from raw transfer measurements.

    This is extracted as a pure function so tests can verify the arithmetic
    independently of any real network transfer, using controlled inputs.

    Formula:
        bps  = bytes_transferred / elapsed_seconds
        mbps = bps * 8 / 1_000_000

    Parameters
    ----------
    bytes_transferred:
        Total bytes of data transferred.
    elapsed_seconds:
        Duration of the transfer in seconds. Must be > 0.

    Returns
    -------
    float
        Throughput in megabits per second.

    Raises
    ------
    ValueError
        If elapsed_seconds <= 0 or bytes_transferred < 0.
    """
    if elapsed_seconds <= 0:
        raise ValueError(f"elapsed_seconds must be > 0, got {elapsed_seconds!r}")
    if bytes_transferred < 0:
        raise ValueError(f"bytes_transferred cannot be negative, got {bytes_transferred!r}")

    bps = bytes_transferred / elapsed_seconds
    return bps * 8 / 1_000_000
