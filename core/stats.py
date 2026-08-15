"""
core/stats.py — RTT statistics and jitter estimation.

This module takes a sequence of already-collected RTT values and computes
descriptive statistics including a jitter estimate.

It performs NO network I/O. It is a pure mathematical transformation of
RTT samples already gathered by run_probes() in core/loss.py.

Jitter definition used in this project
───────────────────────────────────────
We define rtt_jitter_ms as the MEAN ABSOLUTE DIFFERENCE between consecutive
RTT samples:

    Given samples [s0, s1, s2, ..., sN]:
    differences = [|s1-s0|, |s2-s1|, ..., |sN - sN-1|]
    rtt_jitter_ms = mean(differences)

Example:
    samples = [20.0, 22.0, 21.0, 30.0]
    diffs   = [|22-20|, |21-22|, |30-21|]  = [2, 1, 9]
    jitter  = (2 + 1 + 9) / 3              = 4.0 ms

Why this definition?
  - We collect sequential TCP PING/PONG RTT samples. The meaningful question
    is: how much does each measurement differ from the one before it?
    Consecutive absolute differences capture exactly that.
  - It is intuitive and fully explainable without statistical background.
  - It is honest: we call it rtt_jitter_ms and document it explicitly. We
    make no claim to implement RFC 3550 jitter (which requires RTP packet
    arrival timestamps and is specific to real-time streaming protocols).
  - Standard deviation penalizes outliers more (squared deviations) and
    measures spread around the mean rather than sequential variation.
    MAD between consecutive pairs is more natural for sequential probes.

Why NOT RFC 3550 jitter?
  RFC 3550 defines inter-arrival jitter for RTP (real-time streaming)
  packets using their own timestamp fields. It uses an exponential moving
  average (J = J + (|D| - J) / 16) and requires per-packet arrival times
  from the RTP header. Our TCP PING/PONG probes have none of these.

Edge cases
──────────
  0 samples: jitter (and all statistics) are undefined → return None values
  1 sample:  we have a mean/min/max but no consecutive differences exist
             → jitter = 0.0 (zero variation observed, not undefined)
  N samples: normal computation
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RttStats:
    """
    Descriptive statistics computed from a sequence of RTT samples.

    All fields except `count` are None when there are no samples.
    `jitter_ms` is 0.0 (not None) when there is exactly one sample,
    because zero variation is observable — it is a meaningful value.

    Attributes
    ----------
    count:
        Number of RTT samples used to compute these statistics.
    mean_ms:
        Arithmetic mean of all samples in milliseconds. None if count == 0.
    min_ms:
        Minimum (fastest) RTT sample. None if count == 0.
    max_ms:
        Maximum (slowest) RTT sample. None if count == 0.
    jitter_ms:
        RTT variation estimate in milliseconds, defined as the mean absolute
        difference between consecutive RTT samples. None if count == 0.
        0.0 if count == 1 (no consecutive pairs, zero variation observed).
    """

    count: int
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    jitter_ms: float | None

    def __str__(self) -> str:
        if self.count == 0:
            return "RttStats(no samples)"
        return (
            f"RttStats("
            f"n={self.count}, "
            f"mean={self.mean_ms:.2f}ms, "
            f"min={self.min_ms:.2f}ms, "
            f"max={self.max_ms:.2f}ms, "
            f"jitter={self.jitter_ms:.2f}ms"
            f")"
        )


def compute_rtt_stats(samples: tuple[float, ...] | list[float]) -> RttStats:
    """
    Compute descriptive statistics from a sequence of RTT values.

    Parameters
    ----------
    samples:
        RTT measurements in milliseconds. Typically the rtt_values_ms field
        from a ProbeStats object returned by run_probes(). Only successful
        probe RTTs should be included — failed probes have no RTT value.

    Returns
    -------
    RttStats
        Computed statistics. All numeric fields are None if samples is empty.

    Raises
    ------
    ValueError
        If any sample value is negative. A negative RTT is physically
        impossible and indicates a programming error in the caller.
    """
    if not samples:
        return RttStats(count=0, mean_ms=None, min_ms=None, max_ms=None, jitter_ms=None)

    # Validate: negative RTT is physically impossible.
    # This catches caller bugs early (e.g., wrong units, sign error).
    for s in samples:
        if s < 0:
            raise ValueError(
                f"RTT sample cannot be negative, got {s!r}. "
                "Check that samples are in milliseconds and come from successful probes."
            )

    n = len(samples)
    mean_ms = sum(samples) / n
    min_ms = min(samples)
    max_ms = max(samples)

    # Jitter: mean absolute difference between consecutive samples.
    #
    # With only one sample there are no consecutive pairs — there is nothing
    # to compare. We return 0.0 (not None) because the absence of variation
    # is a meaningful result: we observed one RTT and saw no variation.
    if n == 1:
        jitter_ms = 0.0
    else:
        # Compute absolute differences between each adjacent pair.
        # zip(samples, samples[1:]) pairs: (s0,s1), (s1,s2), (s2,s3), ...
        # giving N-1 differences for N samples.
        consecutive_diffs = [
            abs(b - a) for a, b in zip(samples, samples[1:])
        ]
        jitter_ms = sum(consecutive_diffs) / len(consecutive_diffs)

    return RttStats(
        count=n,
        mean_ms=mean_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        jitter_ms=jitter_ms,
    )
