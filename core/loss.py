"""
core/loss.py — Probe loss rate measurement.

This module runs a series of TCP RTT probes against one target and computes
an observed probe loss rate.

Important terminology
─────────────────────
We deliberately do NOT call this "packet loss". Here's why:

  True IP packet loss is measured at the network layer — individual IP packets
  that are dropped by routers, NICs, or firewalls. To observe this directly,
  you need raw sockets or ICMP (which requires elevated permissions).

  Our tcp_rtt_probe() operates at the application layer. TCP hides individual
  packet drops from us by retransmitting them automatically. What we observe
  is whether a complete request/response cycle completed within the timeout.

  Therefore the correct terms are:
    - probe loss rate      (what percentage of our probes failed)
    - probe failure rate   (same)
    - observed loss rate   (what we measured; correlated with, not equal to, IP loss)

  A probe failure can be caused by:
    - True network packet loss (severe enough to exhaust retransmissions)
    - Server crash or restart
    - Our timeout being too short
    - Connection refused (wrong port / service down)
    - DNS failure

  In an interview, say: "We measure probe loss rate — the fraction of our
  application-level probes that don't complete within the timeout. This is
  correlated with true packet loss but is not identical to it."

Loss rate formula
─────────────────
  loss_rate = (failed_probes / total_probes) * 100

  Example: 100 probes, 3 failed → loss_rate = 3.0%

  Edge case: if count=0 or all probes fail, loss_rate is still well-defined
  (0.0 and 100.0 respectively). We validate count > 0 at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.probe import ProbeStatus, RttResult, tcp_rtt_probe


@dataclass(frozen=True)
class ProbeStats:
    """
    Summary statistics from a series of TCP RTT probes to one target.

    Attributes
    ----------
    host:
        Target hostname or IP address.
    port:
        Target port.
    total:
        Total number of probes attempted.
    successful:
        Number of probes that received a valid response (ProbeStatus.SUCCESS).
    failed:
        Number of probes that did not receive a valid response.
        Includes: TIMEOUT, REFUSED, ERROR.
    loss_rate:
        Observed probe loss rate as a percentage [0.0 – 100.0].
        Formula: (failed / total) * 100
        If total == 0, this is 0.0 (caller must prevent this via validation).
    rtt_values_ms:
        RTT values (in milliseconds) from successful probes only.
        Failed probes produce no RTT value.
        This list is preserved so later steps (jitter, statistics) can use it
        without re-running the probes.
    raw_results:
        The full list of RttResult objects, one per probe, in order.
        Preserved for debugging and for future consumers that want the
        full detail beyond the summary statistics.

    Why frozen?
        ProbeStats is a snapshot of a completed measurement session. Mutating
        it after the fact would make results untrustworthy. Immutability also
        makes the object safe to pass around without defensive copying.
    """

    host: str
    port: int
    total: int
    successful: int
    failed: int
    loss_rate: float
    rtt_values_ms: tuple[float, ...]       # tuple not list — immutable, hashable
    raw_results: tuple[RttResult, ...]     # full probe outcomes in order

    def __str__(self) -> str:
        return (
            f"ProbeStats({self.host}:{self.port} | "
            f"total={self.total}, success={self.successful}, failed={self.failed}, "
            f"loss={self.loss_rate:.1f}%)"
        )


def run_probes(
    host: str,
    port: int,
    count: int,
    timeout: float = 2.0,
) -> ProbeStats:
    """
    Run `count` TCP RTT probes against host:port and return summary statistics.

    Each probe is an independent tcp_rtt_probe() call — a fresh TCP connection,
    a PING sent, a PONG expected. Failed probes (any non-SUCCESS status) are
    counted toward the loss rate. Successful probe RTT values are preserved.

    No retry logic: a failed probe counts as failed. Retries would hide
    real failures and complicate the loss rate calculation. This is intentional.

    Parameters
    ----------
    host:
        Target hostname or IP address.
    port:
        TCP port of the echo server.
    count:
        Number of probes to run. Must be >= 1.
    timeout:
        Per-probe timeout in seconds. Passed directly to tcp_rtt_probe().

    Returns
    -------
    ProbeStats
        Summary of all probe outcomes. Never raises — individual probe errors
        are captured in the result rather than propagated to the caller.

    Raises
    ------
    ValueError
        If count < 1. Caller must provide a sensible probe count.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count!r}")

    raw_results: list[RttResult] = []
    rtt_values: list[float] = []

    for _ in range(count):
        result = tcp_rtt_probe(host, port, timeout=timeout)
        raw_results.append(result)

        if result.status == ProbeStatus.SUCCESS and result.rtt_ms is not None:
            rtt_values.append(result.rtt_ms)

    successful = len(rtt_values)
    failed = count - successful
    loss_rate = (failed / count) * 100

    return ProbeStats(
        host=host,
        port=port,
        total=count,
        successful=successful,
        failed=failed,
        loss_rate=loss_rate,
        rtt_values_ms=tuple(rtt_values),
        raw_results=tuple(raw_results),
    )
