"""
core/scoring.py — Deterministic, weighted path scoring and selection engine.

This module is the Step 8 "decision layer". It takes a MonitorResult (the
collection of measured health metrics) and produces a ScoringResult that ranks
all paths by quality and identifies the currently preferred path.

Architecture overview
─────────────────────
Dependency direction is strictly one-way:

    scoring.py (decision layer)
        ↓  reads from
    monitor.py result types (PathMetrics, MonitorResult)
        ↓  which read from
    loss.py, stats.py, throughput.py, path.py (primitives)

The primitives and monitoring layer do NOT know this module exists.
No network I/O happens here — scoring is pure computation over already-collected
data.

Why normalization?
──────────────────
Our four measured metrics have completely different units and scales:
  - RTT:        milliseconds  (typical range: 1 – 500 ms)
  - Probe loss: percentage    (range: 0 – 100 %)
  - Jitter:     milliseconds  (typical range: 0 – 200 ms)
  - Throughput: Mbps          (typical range: 0.1 – 10,000 Mbps)

Adding them directly (e.g., 20ms + 0.5% + 5ms + 150Mbps) produces a number
without physical meaning. The high-magnitude throughput values would dominate
and make RTT and jitter invisible.

Normalization maps every metric onto [0, 100] where 100 always means
"best on this metric across the current set of monitored paths" and 0 means
"worst". After normalization, the scores are comparable and can be combined.

Normalization formulas (relative min-max across the observed set)
─────────────────────────────────────────────────────────────────
Lower-is-better (RTT, loss, jitter):
    score = 100 × (max_value − value) / (max_value − min_value)

    The worst path (highest value) scores 0.
    The best path (lowest value) scores 100.
    Middle paths score proportionally.

Higher-is-better (throughput):
    score = 100 × (value − min_value) / (max_value − min_value)

    The path with the lowest throughput scores 0.
    The path with the highest throughput scores 100.

Edge case — max == min (all paths have the same value on this metric):
    All paths score 50.0 (neutral mid-point).
    Rationale: if every path performs identically on a metric, no path is
    better or worse; it would be arbitrary to score them 0 or 100.
    A mid-point score does not inflate or suppress any path's total score
    on this dimension. This policy is documented in the result.

Availability rule
─────────────────
A path is considered AVAILABLE if and only if:
    probe_stats is not None
    AND probe_stats.loss_rate < 100.0

In other words: if every single probe to that path failed (100% loss), the
path cannot currently communicate. It is marked unavailable.

Why not use throughput failure?
  Throughput measurement involves a separate TCP connection. A path might
  have a functional echo server (so PING probes succeed) but no throughput
  sink server. In that configuration, declaring the path unavailable because
  throughput failed would be wrong. The essential reachability signal is
  whether any probe round-trip succeeded at all.

Why not check for rtt_stats?
  rtt_stats is derived from probe_stats.rtt_values_ms. If any probes succeed,
  rtt_stats.count > 0. The probe_stats loss check already covers this.

Unavailable path treatment:
  - Marked available=False
  - Still included in the result (the result remains inspectable)
  - Normalized metric scores are set to 0.0 for all metrics
  - Weighted score is set to 0.0
  - Cannot be the preferred path; only available paths are ranked first

Weighted scoring
────────────────
After normalization, the total score is:

    total_score = (rtt_score   × w_rtt)
               + (loss_score  × w_loss)
               + (jitter_score × w_jitter)
               + (throughput_score × w_throughput)

Weights represent the relative importance of each metric as a policy decision.
Default: RTT=0.30, loss=0.30, jitter=0.20, throughput=0.20.
These are NOT universal networking standards — they reflect the operator's
traffic priorities. A VoIP operator might weight jitter more heavily.

Weight validation:
  - All weights must be >= 0.0
  - They must sum to 1.0 (within floating-point tolerance of 1e-9)
  - At least one weight must be > 0.0

Why require sum == 1.0?
  This makes total_score directly interpretable as a 0–100 value where 100
  means "best possible on every metric". If weights summed to e.g. 0.5, the
  best possible score would be 50, making comparisons confusing. If they
  summed to 2.0, scores could exceed 100. Requiring sum == 1.0 keeps the
  output range well-defined and auditable.

Tie handling
────────────
If two paths have the same total_score (floating-point equal), the tie is
broken by path.name in ascending lexicographic order. This is:
  - Deterministic (always the same result for the same input)
  - Operator-visible (the operator chose the path names)
  - Simple to explain and audit

No randomness is introduced.

Missing metric handling
───────────────────────
When a metric is missing from a PathMetrics (e.g., all probes failed so
rtt_stats.count == 0, or throughput timed out so throughput_mbps is None),
the scoring engine treats the raw value as None. During normalization, a path
with a missing metric value is excluded from computing the min/max range for
that metric, and receives a score of 0.0 on that metric. This penalizes
missing metrics without distorting the scores of paths that have data.

Exception: if ALL paths have a missing value for some metric (e.g., no path
has working throughput), that metric contributes 0.0 to every path's score.
The weight remains — the metric simply contributes nothing because there is
nothing to compare. The scoring result documents this via the score field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.monitor import MonitorResult, PathMetrics
from core.path import Path

# ── Default weights ────────────────────────────────────────────────────────────

DEFAULT_WEIGHT_RTT: float = 0.30
DEFAULT_WEIGHT_LOSS: float = 0.30
DEFAULT_WEIGHT_JITTER: float = 0.20
DEFAULT_WEIGHT_THROUGHPUT: float = 0.20

# Floating-point tolerance for weight-sum validation.
# 0.30 + 0.30 + 0.20 + 0.20 in binary floating-point may not be exactly 1.0.
_WEIGHT_SUM_TOLERANCE: float = 1e-9

# Score assigned when all paths have the same value for a metric (no spread).
# Mid-point: no path is penalised or rewarded relative to the others.
_EQUAL_METRIC_SCORE: float = 50.0

# Score assigned to a path with a missing metric value.
_MISSING_METRIC_SCORE: float = 0.0


# ── Weight configuration ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoringWeights:
    """
    Configurable importance weights for the four measured metrics.

    Each weight is the fraction of the total score that metric contributes.
    They must be non-negative and sum to 1.0.

    Attributes
    ----------
    rtt:
        Weight for normalized RTT score. Default: 0.30.
    loss:
        Weight for normalized probe-loss score. Default: 0.30.
    jitter:
        Weight for normalized jitter score. Default: 0.20.
    throughput:
        Weight for normalized throughput score. Default: 0.20.

    Why configurable?
        Different traffic types have different sensitivities:
          - VoIP / video: prioritize low RTT and low jitter
          - Bulk upload:  prioritize throughput
          - API gateway:  prioritize low loss and low RTT
        Hard-coding weights would make the engine inflexible.

    Example
    -------
        # VoIP-optimized weights
        voip_weights = ScoringWeights(rtt=0.35, loss=0.35, jitter=0.25, throughput=0.05)
    """

    rtt: float = DEFAULT_WEIGHT_RTT
    loss: float = DEFAULT_WEIGHT_LOSS
    jitter: float = DEFAULT_WEIGHT_JITTER
    throughput: float = DEFAULT_WEIGHT_THROUGHPUT

    def __post_init__(self) -> None:
        """
        Validate all weights at construction time.

        Raises
        ------
        ValueError
            If any weight is negative, or if weights do not sum to 1.0
            (within floating-point tolerance).
        """
        for name, value in [
            ("rtt", self.rtt),
            ("loss", self.loss),
            ("jitter", self.jitter),
            ("throughput", self.throughput),
        ]:
            if value < 0.0:
                raise ValueError(
                    f"Weight '{name}' must be >= 0.0, got {value!r}. "
                    "A negative weight has no meaningful interpretation."
                )

        total = self.rtt + self.loss + self.jitter + self.throughput
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.10f}. "
                "This ensures total_score stays in the range [0, 100] and "
                "is directly comparable across different weight configurations."
            )

        if total == 0.0:
            # This is caught by the sum check above (0.0 != 1.0), but documented
            # explicitly for clarity.
            raise ValueError("At least one weight must be > 0.0.")


# ── Per-path scored result ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredPath:
    """
    The scoring result for a single path.

    Contains the original PathMetrics plus all intermediate and final score
    values, so callers can understand EXACTLY how the score was computed.
    The result is intentionally inspectable — not just a final number.

    Attributes
    ----------
    path:
        The Path configuration that was scored.
    metrics:
        The raw PathMetrics produced by the monitoring layer. Included so
        callers can trace back from a score to the underlying measurements.
    available:
        True if the path can currently communicate (probe_stats.loss_rate < 100).
        Unavailable paths score 0.0 on every metric and cannot be preferred.
    rtt_score:
        Normalized RTT score in [0, 100]. Higher = better RTT (lower raw RTT).
        0.0 if the path is unavailable or has no RTT data.
    loss_score:
        Normalized probe-loss score in [0, 100]. Higher = better (lower loss).
        0.0 if the path is unavailable.
    jitter_score:
        Normalized jitter score in [0, 100]. Higher = better (lower jitter).
        0.0 if the path is unavailable or has no jitter data.
    throughput_score:
        Normalized throughput score in [0, 100]. Higher = better (more throughput).
        0.0 if the path is unavailable or throughput measurement failed.
    total_score:
        Weighted sum of the four normalized scores.
        Range: [0, 100] for available paths.
        Always 0.0 for unavailable paths.
    rank:
        Position in the final ranking (1 = best). Available paths are ranked
        above unavailable paths. Within a group, ranked by total_score descending,
        with ties broken by path.name ascending (lexicographic).
    """

    path: Path
    metrics: PathMetrics
    available: bool
    rtt_score: float
    loss_score: float
    jitter_score: float
    throughput_score: float
    total_score: float
    rank: int

    def __str__(self) -> str:
        avail = "available" if self.available else "UNAVAILABLE"
        return (
            f"ScoredPath(rank={self.rank}, name={self.path.name!r}, "
            f"{avail}, total={self.total_score:.2f}, "
            f"rtt={self.rtt_score:.1f}, loss={self.loss_score:.1f}, "
            f"jitter={self.jitter_score:.1f}, tput={self.throughput_score:.1f})"
        )


# ── Final scoring result ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoringResult:
    """
    The complete output of the scoring engine for a set of paths.

    Attributes
    ----------
    scored_paths:
        All paths, scored and ranked. Includes unavailable paths (marked as such).
        Ordered by rank ascending (rank 1 = preferred path is first).
    preferred_path:
        The top-ranked available path, or None if no path is available.
        This is the path the operator would use if implementing path selection.
        It is NOT automatically used for routing — this project does not change
        OS routing tables, create tunnels, or move packets.
    weights:
        The ScoringWeights configuration used to produce this result.
        Included for auditability.

    Why include unavailable paths?
        Hiding failed paths from the result would make it harder to diagnose
        problems. An operator viewing the result should see ALL paths and their
        status, not just the healthy ones.

    Why separate preferred_path?
        Callers should be able to ask "what is the best path right now?" with
        a single attribute access, not by iterating scored_paths.

    Why include weights in the result?
        The same MonitorResult might be scored with different weight
        configurations. Including the weights used makes the result self-
        describing and reproducible.
    """

    scored_paths: list[ScoredPath]
    preferred_path: ScoredPath | None
    weights: ScoringWeights

    def __str__(self) -> str:
        if not self.scored_paths:
            return "ScoringResult(no paths)"
        preferred_name = (
            self.preferred_path.path.name if self.preferred_path else None
        )
        lines = [
            f"ScoringResult({len(self.scored_paths)} paths, "
            f"preferred={preferred_name!r}):"
        ]
        for sp in self.scored_paths:
            lines.append(f"  {sp}")
        return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _normalize_lower_is_better(
    value: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Normalize a lower-is-better metric to [0, 100].

    Formula: 100 × (max_val − value) / (max_val − min_val)

    When max_val == min_val (all paths are equal): returns _EQUAL_METRIC_SCORE.
    This avoids a division-by-zero and assigns a neutral mid-point score.

    Parameters
    ----------
    value:
        The raw metric value to normalize.
    min_val:
        Minimum observed value across all available paths with this metric.
    max_val:
        Maximum observed value across all available paths with this metric.

    Returns
    -------
    float
        Normalized score in [0.0, 100.0].
    """
    if max_val == min_val:
        return _EQUAL_METRIC_SCORE
    return 100.0 * (max_val - value) / (max_val - min_val)


def _normalize_higher_is_better(
    value: float,
    min_val: float,
    max_val: float,
) -> float:
    """
    Normalize a higher-is-better metric to [0, 100].

    Formula: 100 × (value − min_val) / (max_val − min_val)

    When max_val == min_val (all paths are equal): returns _EQUAL_METRIC_SCORE.

    Parameters
    ----------
    value:
        The raw metric value to normalize.
    min_val:
        Minimum observed value across all available paths with this metric.
    max_val:
        Maximum observed value across all available paths with this metric.

    Returns
    -------
    float
        Normalized score in [0.0, 100.0].
    """
    if max_val == min_val:
        return _EQUAL_METRIC_SCORE
    return 100.0 * (value - min_val) / (max_val - min_val)


def _extract_raw_metrics(
    pm: PathMetrics,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Extract the four raw metric values from a PathMetrics.

    Returns (rtt_ms, loss_pct, jitter_ms, throughput_mbps).
    Any value may be None if the metric was not collected or all probes failed.

    This is a pure extraction function — no computation, no side effects.
    It centralises the logic of "where does each metric live in PathMetrics?"
    so the rest of the scoring code doesn't repeat the same attribute paths.
    """
    rtt_ms: float | None = None
    loss_pct: float | None = None
    jitter_ms: float | None = None
    throughput_mbps: float | None = None

    if pm.probe_stats is not None:
        loss_pct = pm.probe_stats.loss_rate

    if (
        pm.rtt_stats is not None
        and pm.rtt_stats.count > 0
        and pm.rtt_stats.mean_ms is not None
    ):
        rtt_ms = pm.rtt_stats.mean_ms

    if (
        pm.rtt_stats is not None
        and pm.rtt_stats.count > 0
        and pm.rtt_stats.jitter_ms is not None
    ):
        jitter_ms = pm.rtt_stats.jitter_ms

    if pm.throughput is not None and pm.throughput.throughput_mbps is not None:
        throughput_mbps = pm.throughput.throughput_mbps

    return rtt_ms, loss_pct, jitter_ms, throughput_mbps


def _is_available(pm: PathMetrics) -> bool:
    """
    Determine whether a path is currently available.

    A path is AVAILABLE if:
        pm.probe_stats is not None
        AND pm.probe_stats.loss_rate < 100.0

    This means at least one probe successfully completed a round-trip.

    A path is UNAVAILABLE if:
        - probe_stats is None (monitoring could not run at all)
        - loss_rate == 100.0 (every single probe failed)

    We use probe loss as the availability signal because:
      1. Probes are the fundamental reachability test — if no PING returns
         a PONG, the path is by definition unreachable to our measurement.
      2. Throughput failure could mean the throughput server is misconfigured,
         not that the path is down. Using throughput would cause false negatives.
      3. 100% loss is an unambiguous signal that all connectivity attempts failed.

    The threshold is strictly < 100.0. A path with 99% loss is still technically
    reachable (1% of probes got through), though it will score very poorly on
    the loss metric.
    """
    return (
        pm.probe_stats is not None
        and pm.probe_stats.loss_rate < 100.0
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def score_paths(
    monitor_result: MonitorResult,
    weights: ScoringWeights | None = None,
) -> ScoringResult:
    """
    Score and rank all paths in a MonitorResult.

    This function:
      1. Determines which paths are available (probe loss < 100%).
      2. Collects raw metric values from available paths.
      3. Computes min/max ranges for each metric across available paths.
      4. Normalizes each metric to [0, 100].
      5. Computes a weighted total score for each path.
      6. Ranks paths: available paths first (sorted by total_score desc),
         unavailable paths after (sorted by name asc — no score to sort by).
      7. Identifies the preferred path (rank 1 if available; None if no
         available path exists).

    Parameters
    ----------
    monitor_result:
        The result of a monitoring run (from monitor_paths()). May be empty.
    weights:
        Scoring weights. If None, uses the default ScoringWeights.
        Must be a valid ScoringWeights instance (weights sum to 1.0,
        all non-negative).

    Returns
    -------
    ScoringResult
        Always returns — never raises on empty or failed paths.

    Raises
    ------
    TypeError
        If weights is not a ScoringWeights instance (and not None).

    Notes
    ─────
    Unavailable paths (100% loss or no probe_stats) are scored 0.0 on every
    metric and assigned ranks after all available paths.

    If no paths are available, preferred_path is None.
    If no paths are configured at all, scored_paths is [] and preferred_path
    is None.

    Numerical example with default weights (RTT=0.30, loss=0.30, jitter=0.20,
    throughput=0.20):

      Path A: RTT=10ms, loss=0%, jitter=1ms, throughput=100Mbps
      Path B: RTT=50ms, loss=5%, jitter=9ms, throughput=20Mbps

      RTT normalization (lower is better):
        min=10, max=50
        A: 100×(50-10)/(50-10) = 100.0
        B: 100×(50-50)/(50-10) = 0.0

      Loss normalization (lower is better):
        min=0, max=5
        A: 100×(5-0)/(5-0) = 100.0
        B: 100×(5-5)/(5-0) = 0.0

      Jitter normalization (lower is better):
        min=1, max=9
        A: 100×(9-1)/(9-1) = 100.0
        B: 100×(9-9)/(9-1) = 0.0

      Throughput normalization (higher is better):
        min=20, max=100
        A: 100×(100-20)/(100-20) = 100.0
        B: 100×(20-20)/(100-20) = 0.0

      Total:
        A: 100×0.30 + 100×0.30 + 100×0.20 + 100×0.20 = 100.0
        B: 0×0.30 + 0×0.30 + 0×0.20 + 0×0.20 = 0.0
    """
    if weights is None:
        weights = ScoringWeights()

    all_metrics: dict[str, PathMetrics] = monitor_result.metrics

    if not all_metrics:
        return ScoringResult(scored_paths=[], preferred_path=None, weights=weights)

    # ── Step 1: Determine availability and extract raw values ─────────────────

    # available_raw: name → (rtt, loss, jitter, throughput) — only available paths
    available_raw: dict[str, tuple[float | None, float | None, float | None, float | None]] = {}
    unavailable_names: list[str] = []

    for name, pm in all_metrics.items():
        if _is_available(pm):
            available_raw[name] = _extract_raw_metrics(pm)
        else:
            unavailable_names.append(name)

    # ── Step 2: Compute min/max ranges for each metric (available paths only) ──
    #
    # Only available paths contribute to the normalization range.
    # Including unavailable paths would distort the range: a completely dead
    # path with no RTT data would contribute a None, which we'd have to exclude
    # anyway, and a 100% loss value would pull the loss range to 100 even for
    # a healthy set of paths.

    rtt_values:        list[float] = []
    loss_values:       list[float] = []
    jitter_values:     list[float] = []
    throughput_values: list[float] = []

    for rtt, loss, jitter, tput in available_raw.values():
        if rtt        is not None: rtt_values.append(rtt)
        if loss       is not None: loss_values.append(loss)
        if jitter     is not None: jitter_values.append(jitter)
        if tput       is not None: throughput_values.append(tput)

    rtt_min,        rtt_max        = (min(rtt_values),        max(rtt_values))        if rtt_values        else (0.0, 0.0)
    loss_min,       loss_max       = (min(loss_values),       max(loss_values))       if loss_values       else (0.0, 0.0)
    jitter_min,     jitter_max     = (min(jitter_values),     max(jitter_values))     if jitter_values     else (0.0, 0.0)
    throughput_min, throughput_max = (min(throughput_values), max(throughput_values)) if throughput_values else (0.0, 0.0)

    # ── Step 3: Score available paths ─────────────────────────────────────────

    available_scored: list[ScoredPath] = []

    for name, pm in all_metrics.items():
        if name in unavailable_names:
            continue   # handled in step 4

        rtt, loss, jitter, tput = available_raw[name]

        rtt_score = (
            _normalize_lower_is_better(rtt, rtt_min, rtt_max)
            if rtt is not None
            else _MISSING_METRIC_SCORE
        )
        loss_score = (
            _normalize_lower_is_better(loss, loss_min, loss_max)
            if loss is not None
            else _MISSING_METRIC_SCORE
        )
        jitter_score = (
            _normalize_lower_is_better(jitter, jitter_min, jitter_max)
            if jitter is not None
            else _MISSING_METRIC_SCORE
        )
        throughput_score = (
            _normalize_higher_is_better(tput, throughput_min, throughput_max)
            if tput is not None
            else _MISSING_METRIC_SCORE
        )

        total_score = (
            rtt_score        * weights.rtt
            + loss_score     * weights.loss
            + jitter_score   * weights.jitter
            + throughput_score * weights.throughput
        )

        available_scored.append(
            ScoredPath(
                path=pm.path,
                metrics=pm,
                available=True,
                rtt_score=rtt_score,
                loss_score=loss_score,
                jitter_score=jitter_score,
                throughput_score=throughput_score,
                total_score=total_score,
                rank=0,  # assigned below after sorting
            )
        )

    # ── Step 4: Sort available paths — score descending, name ascending on tie ─
    #
    # Tie-breaking by name (ascending lexicographic) is:
    #   - Deterministic: always the same result for the same input
    #   - Operator-visible: the operator chose the path names
    #   - No randomness introduced
    available_scored.sort(key=lambda sp: (-sp.total_score, sp.path.name))

    # ── Step 5: Score unavailable paths (all zeros, sorted by name) ───────────
    unavailable_scored: list[ScoredPath] = []
    for name in sorted(unavailable_names):  # stable sort by name
        pm = all_metrics[name]
        unavailable_scored.append(
            ScoredPath(
                path=pm.path,
                metrics=pm,
                available=False,
                rtt_score=0.0,
                loss_score=0.0,
                jitter_score=0.0,
                throughput_score=0.0,
                total_score=0.0,
                rank=0,  # assigned below
            )
        )

    # ── Step 6: Combine and assign ranks ──────────────────────────────────────
    #
    # Available paths come first (ranks 1..N), unavailable paths after.
    all_scored: list[ScoredPath] = []
    for rank, sp in enumerate(available_scored + unavailable_scored, start=1):
        # ScoredPath is frozen — rebuild with rank assigned
        all_scored.append(
            ScoredPath(
                path=sp.path,
                metrics=sp.metrics,
                available=sp.available,
                rtt_score=sp.rtt_score,
                loss_score=sp.loss_score,
                jitter_score=sp.jitter_score,
                throughput_score=sp.throughput_score,
                total_score=sp.total_score,
                rank=rank,
            )
        )

    # ── Step 7: Identify preferred path ───────────────────────────────────────
    preferred_path: ScoredPath | None = None
    if all_scored and all_scored[0].available:
        preferred_path = all_scored[0]

    return ScoringResult(
        scored_paths=all_scored,
        preferred_path=preferred_path,
        weights=weights,
    )
