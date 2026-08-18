"""
tests/test_scoring.py — Deterministic unit tests for core/scoring.py.

Test strategy
─────────────
All tests use manually constructed PathMetrics objects — no real network I/O.
Every expected score is hand-calculated from the normalization formulas so the
test is self-documenting and auditable:

  Lower-is-better formula:
      score = 100 × (max_value − value) / (max_value − min_value)

  Higher-is-better formula:
      score = 100 × (value − min_value) / (max_value − min_value)

  max == min edge case → score = 50.0 (neutral mid-point)

We use pytest.approx() for floating-point comparisons (tolerance 1e-6)
because IEEE 754 arithmetic may produce tiny rounding errors.

What is tested
──────────────
 Part 1: ScoringWeights validation
 Part 2: Normalization helper functions directly
 Part 3: Single path scoring (degenerate case — max == min for all metrics)
 Part 4: Two-path scoring with hand-calculated expected values
 Part 5: Availability rule (100% loss = unavailable)
 Part 6: Failed paths cannot be preferred
 Part 7: Ranking with multiple paths
 Part 8: Tie handling (same score → alphabetical by name)
 Part 9: Custom weights
 Part 10: Missing metrics
 Part 11: Empty path input
 Part 12: ScoringResult is inspectable (preferred_path, scored_paths, weights)

Fixtures
────────
We use helper functions rather than pytest fixtures for constructing
PathMetrics, because each test needs slightly different values. Small
factory functions keep the setup explicit and the expected values obvious.
"""

from __future__ import annotations

import pytest

from core.monitor import MonitorResult, PathMetrics
from core.path import Path
from core.probe import ProbeStatus
from core.scoring import (
    ScoredPath,
    ScoringResult,
    ScoringWeights,
    _EQUAL_METRIC_SCORE,
    _MISSING_METRIC_SCORE,
    _normalize_higher_is_better,
    _normalize_lower_is_better,
    score_paths,
)

# We also import result types used to build PathMetrics
from core.loss import ProbeStats
from core.stats import RttStats
from core.throughput import ThroughputResult

# ── Builder helpers ────────────────────────────────────────────────────────────


def make_path(name: str = "test") -> Path:
    """Create a Path with dummy host/port — values don't matter for scoring."""
    return Path(name=name, host="127.0.0.1", port=9000)


def make_probe_stats(
    host: str = "127.0.0.1",
    port: int = 9000,
    total: int = 5,
    successful: int = 5,
    rtt_values: tuple[float, ...] = (10.0, 10.0, 10.0, 10.0, 10.0),
) -> ProbeStats:
    """
    Build a ProbeStats with controlled values.

    failed = total - successful
    loss_rate = (failed / total) * 100
    """
    failed = total - successful
    loss_rate = (failed / total) * 100 if total > 0 else 0.0
    return ProbeStats(
        host=host,
        port=port,
        total=total,
        successful=successful,
        failed=failed,
        loss_rate=loss_rate,
        rtt_values_ms=rtt_values,
        raw_results=(),
    )


def make_rtt_stats(
    mean_ms: float = 10.0,
    min_ms: float = 9.0,
    max_ms: float = 11.0,
    jitter_ms: float = 1.0,
    count: int = 5,
) -> RttStats:
    """Build an RttStats with controlled values."""
    return RttStats(
        count=count,
        mean_ms=mean_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        jitter_ms=jitter_ms,
    )


def make_throughput_result(
    host: str = "127.0.0.1",
    port: int = 9000,
    throughput_mbps: float | None = 100.0,
    status: ProbeStatus = ProbeStatus.SUCCESS,
) -> ThroughputResult:
    """Build a ThroughputResult with controlled values."""
    if throughput_mbps is not None:
        bytes_transferred = 1024 * 1024
        elapsed_seconds = (bytes_transferred * 8) / (throughput_mbps * 1_000_000)
        throughput_bps = throughput_mbps * 1_000_000 / 8
    else:
        bytes_transferred = None
        elapsed_seconds = None
        throughput_bps = None

    return ThroughputResult(
        host=host,
        port=port,
        status=status,
        bytes_transferred=bytes_transferred,
        elapsed_seconds=elapsed_seconds,
        throughput_bps=throughput_bps,
        throughput_mbps=throughput_mbps,
        error_message=None if status == ProbeStatus.SUCCESS else "connection refused",
    )


def make_path_metrics(
    name: str = "test",
    rtt_mean_ms: float = 10.0,
    jitter_ms: float = 1.0,
    loss_pct: float = 0.0,
    throughput_mbps: float | None = 100.0,
    total_probes: int = 5,
    throughput_status: ProbeStatus = ProbeStatus.SUCCESS,
) -> PathMetrics:
    """
    Build a complete PathMetrics for one path with controlled metric values.

    successful probes = round(total_probes × (1 - loss_pct/100))
    """
    path = make_path(name)
    successful = round(total_probes * (1 - loss_pct / 100))
    successful = max(0, min(total_probes, successful))

    rtt_vals = tuple(rtt_mean_ms for _ in range(successful))

    probe_stats = make_probe_stats(
        host=path.host,
        port=path.port,
        total=total_probes,
        successful=successful,
        rtt_values=rtt_vals,
    )

    if successful > 0:
        rtt_stats = make_rtt_stats(
            mean_ms=rtt_mean_ms,
            min_ms=rtt_mean_ms,
            max_ms=rtt_mean_ms,
            jitter_ms=jitter_ms,
            count=successful,
        )
    else:
        # No successful probes → empty RttStats
        rtt_stats = RttStats(count=0, mean_ms=None, min_ms=None, max_ms=None, jitter_ms=None)

    throughput = make_throughput_result(
        host=path.host,
        port=path.port,
        throughput_mbps=throughput_mbps if throughput_status == ProbeStatus.SUCCESS else None,
        status=throughput_status,
    )

    return PathMetrics(
        path=path,
        probe_stats=probe_stats,
        rtt_stats=rtt_stats,
        throughput=throughput,
    )


def make_monitor_result(*path_metrics_list: PathMetrics) -> MonitorResult:
    """Build a MonitorResult from one or more PathMetrics."""
    return MonitorResult(
        metrics={pm.path.name: pm for pm in path_metrics_list}
    )


def make_unavailable_path_metrics(name: str = "dead") -> PathMetrics:
    """
    Build a PathMetrics where every probe failed (100% loss).
    This path will be marked unavailable by the scoring engine.
    """
    path = make_path(name)
    probe_stats = make_probe_stats(
        host=path.host,
        port=path.port,
        total=5,
        successful=0,
        rtt_values=(),
    )
    rtt_stats = RttStats(count=0, mean_ms=None, min_ms=None, max_ms=None, jitter_ms=None)
    throughput = make_throughput_result(
        host=path.host,
        port=path.port,
        throughput_mbps=None,
        status=ProbeStatus.REFUSED,
    )
    return PathMetrics(
        path=path,
        probe_stats=probe_stats,
        rtt_stats=rtt_stats,
        throughput=throughput,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: ScoringWeights validation
# ══════════════════════════════════════════════════════════════════════════════


class TestScoringWeights:
    """ScoringWeights must validate inputs at construction time."""

    def test_default_weights_sum_to_one(self) -> None:
        w = ScoringWeights()
        assert abs(w.rtt + w.loss + w.jitter + w.throughput - 1.0) < 1e-9

    def test_default_weights_values(self) -> None:
        w = ScoringWeights()
        assert w.rtt == pytest.approx(0.30)
        assert w.loss == pytest.approx(0.30)
        assert w.jitter == pytest.approx(0.20)
        assert w.throughput == pytest.approx(0.20)

    def test_custom_weights_valid(self) -> None:
        w = ScoringWeights(rtt=0.50, loss=0.25, jitter=0.15, throughput=0.10)
        assert w.rtt == pytest.approx(0.50)
        assert w.loss == pytest.approx(0.25)

    def test_negative_rtt_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="rtt"):
            ScoringWeights(rtt=-0.1, loss=0.4, jitter=0.4, throughput=0.3)

    def test_negative_loss_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="loss"):
            ScoringWeights(rtt=0.5, loss=-0.1, jitter=0.4, throughput=0.2)

    def test_negative_jitter_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            ScoringWeights(rtt=0.4, loss=0.4, jitter=-0.1, throughput=0.3)

    def test_negative_throughput_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="throughput"):
            ScoringWeights(rtt=0.4, loss=0.4, jitter=0.3, throughput=-0.1)

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            ScoringWeights(rtt=0.50, loss=0.50, jitter=0.10, throughput=0.10)

    def test_all_zero_weights_raise(self) -> None:
        """All-zero weights do not sum to 1.0, so they are rejected."""
        with pytest.raises(ValueError):
            ScoringWeights(rtt=0.0, loss=0.0, jitter=0.0, throughput=0.0)

    def test_weights_immutable(self) -> None:
        """ScoringWeights is frozen — attribute assignment must raise."""
        w = ScoringWeights()
        with pytest.raises(Exception):
            w.rtt = 0.5  # type: ignore[misc]

    def test_zero_weight_for_one_metric_valid(self) -> None:
        """A single zero weight is valid as long as others sum to 1.0."""
        w = ScoringWeights(rtt=0.40, loss=0.40, jitter=0.20, throughput=0.0)
        assert w.throughput == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Normalization helper functions
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizationHelpers:
    """
    Unit tests for _normalize_lower_is_better and _normalize_higher_is_better.

    All expected values are hand-calculated from the formulas:
      lower-is-better: 100 × (max − value) / (max − min)
      higher-is-better: 100 × (value − min) / (max − min)
    """

    # ── lower-is-better ───────────────────────────────────────────────────────

    def test_lower_is_better_minimum_value_scores_100(self) -> None:
        """
        The best (lowest) value always scores 100.
          value=10, min=10, max=50 → 100×(50-10)/(50-10) = 100.0
        """
        assert _normalize_lower_is_better(10.0, 10.0, 50.0) == pytest.approx(100.0)

    def test_lower_is_better_maximum_value_scores_0(self) -> None:
        """
        The worst (highest) value always scores 0.
          value=50, min=10, max=50 → 100×(50-50)/(50-10) = 0.0
        """
        assert _normalize_lower_is_better(50.0, 10.0, 50.0) == pytest.approx(0.0)

    def test_lower_is_better_midpoint(self) -> None:
        """
        A value exactly midway between min and max scores 50.
          value=30, min=10, max=50 → 100×(50-30)/(50-10) = 50.0
        """
        assert _normalize_lower_is_better(30.0, 10.0, 50.0) == pytest.approx(50.0)

    def test_lower_is_better_specific_value(self) -> None:
        """
        Hand-calculated example from the requirements:
          min=10, max=50, value=20
          score = 100 × (50-20)/(50-10) = 100×30/40 = 75.0
        """
        assert _normalize_lower_is_better(20.0, 10.0, 50.0) == pytest.approx(75.0)

    def test_lower_is_better_equal_min_max_returns_midpoint(self) -> None:
        """
        When all paths have the same value (max == min), return 50.0.
        This is the equal-competitor neutral mid-point policy.
        """
        assert _normalize_lower_is_better(42.0, 42.0, 42.0) == pytest.approx(_EQUAL_METRIC_SCORE)

    def test_lower_is_better_near_minimum(self) -> None:
        """
        value=11, min=10, max=50 → 100×(50-11)/(50-10) = 100×39/40 = 97.5
        """
        assert _normalize_lower_is_better(11.0, 10.0, 50.0) == pytest.approx(97.5)

    # ── higher-is-better ──────────────────────────────────────────────────────

    def test_higher_is_better_maximum_value_scores_100(self) -> None:
        """
        The best (highest) value scores 100.
          value=100, min=20, max=100 → 100×(100-20)/(100-20) = 100.0
        """
        assert _normalize_higher_is_better(100.0, 20.0, 100.0) == pytest.approx(100.0)

    def test_higher_is_better_minimum_value_scores_0(self) -> None:
        """
        The worst (lowest) value scores 0.
          value=20, min=20, max=100 → 100×(20-20)/(100-20) = 0.0
        """
        assert _normalize_higher_is_better(20.0, 20.0, 100.0) == pytest.approx(0.0)

    def test_higher_is_better_midpoint(self) -> None:
        """
        value=60, min=20, max=100 → 100×(60-20)/(100-20) = 50.0
        """
        assert _normalize_higher_is_better(60.0, 20.0, 100.0) == pytest.approx(50.0)

    def test_higher_is_better_specific_value(self) -> None:
        """
        value=50, min=20, max=100 → 100×(50-20)/(100-20) = 100×30/80 = 37.5
        """
        assert _normalize_higher_is_better(50.0, 20.0, 100.0) == pytest.approx(37.5)

    def test_higher_is_better_equal_min_max_returns_midpoint(self) -> None:
        """When all paths have the same throughput, return 50.0."""
        assert _normalize_higher_is_better(77.0, 77.0, 77.0) == pytest.approx(_EQUAL_METRIC_SCORE)


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Single path scoring (degenerate max == min case)
# ══════════════════════════════════════════════════════════════════════════════


class TestSinglePath:
    """
    With only one path, every metric has max == min. Each normalized score is
    50.0 (the equal-competitor neutral mid-point). Total score = 50.0.
    """

    def test_single_path_returns_scoring_result(self) -> None:
        pm = make_path_metrics("only")
        result = score_paths(make_monitor_result(pm))
        assert isinstance(result, ScoringResult)

    def test_single_path_is_preferred(self) -> None:
        pm = make_path_metrics("only")
        result = score_paths(make_monitor_result(pm))
        assert result.preferred_path is not None
        assert result.preferred_path.path.name == "only"

    def test_single_path_rank_is_1(self) -> None:
        pm = make_path_metrics("only")
        result = score_paths(make_monitor_result(pm))
        assert result.preferred_path.rank == 1

    def test_single_path_all_component_scores_are_midpoint(self) -> None:
        """
        One path → max == min for all metrics → each score == 50.0.
        """
        pm = make_path_metrics("only")
        result = score_paths(make_monitor_result(pm))
        sp = result.scored_paths[0]
        assert sp.rtt_score        == pytest.approx(_EQUAL_METRIC_SCORE)
        assert sp.loss_score       == pytest.approx(_EQUAL_METRIC_SCORE)
        assert sp.jitter_score     == pytest.approx(_EQUAL_METRIC_SCORE)
        assert sp.throughput_score == pytest.approx(_EQUAL_METRIC_SCORE)

    def test_single_path_total_score(self) -> None:
        """
        total = 50×0.30 + 50×0.30 + 50×0.20 + 50×0.20
              = 15 + 15 + 10 + 10 = 50.0
        """
        pm = make_path_metrics("only")
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].total_score == pytest.approx(50.0)

    def test_single_path_available(self) -> None:
        pm = make_path_metrics("only")
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].available is True


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Two-path scoring — exact hand-calculated values
# ══════════════════════════════════════════════════════════════════════════════


class TestTwoPathScoring:
    """
    Two paths with distinct metrics. All expected values are derived by hand
    from the normalization formulas. Using default weights (0.30/0.30/0.20/0.20).

    Path A (good):  rtt=10ms, loss=0%,  jitter=1ms,  throughput=100Mbps
    Path B (poor):  rtt=50ms, loss=5%,  jitter=9ms,  throughput=20Mbps

    ── RTT normalization (lower is better) ──────────────────────────
      min=10, max=50
      A: 100×(50-10)/(50-10) = 100.0
      B: 100×(50-50)/(50-10) =   0.0

    ── Loss normalization (lower is better) ─────────────────────────
      min=0, max=5
      A: 100×(5-0)/(5-0)     = 100.0
      B: 100×(5-5)/(5-0)     =   0.0

    ── Jitter normalization (lower is better) ───────────────────────
      min=1, max=9
      A: 100×(9-1)/(9-1)     = 100.0
      B: 100×(9-9)/(9-1)     =   0.0

    ── Throughput normalization (higher is better) ───────────────────
      min=20, max=100
      A: 100×(100-20)/(100-20) = 100.0
      B: 100×( 20-20)/(100-20) =   0.0

    ── Total scores (weights 0.30/0.30/0.20/0.20) ───────────────────
      A: 100×0.30 + 100×0.30 + 100×0.20 + 100×0.20 = 100.0
      B:   0×0.30 +   0×0.30 +   0×0.20 +   0×0.20 =   0.0
    """

    @pytest.fixture()
    def two_path_result(self) -> ScoringResult:
        pm_a = make_path_metrics(
            name="alpha", rtt_mean_ms=10.0, jitter_ms=1.0,
            loss_pct=0.0, throughput_mbps=100.0,
        )
        pm_b = make_path_metrics(
            name="beta", rtt_mean_ms=50.0, jitter_ms=9.0,
            loss_pct=5.0, throughput_mbps=20.0,
            total_probes=20,  # need enough to get ~5% loss
        )
        return score_paths(make_monitor_result(pm_a, pm_b))

    def test_alpha_rtt_score(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "alpha")
        assert sp.rtt_score == pytest.approx(100.0)

    def test_beta_rtt_score(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "beta")
        assert sp.rtt_score == pytest.approx(0.0)

    def test_alpha_total_score(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "alpha")
        assert sp.total_score == pytest.approx(100.0)

    def test_beta_total_score(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "beta")
        assert sp.total_score == pytest.approx(0.0)

    def test_alpha_is_preferred(self, two_path_result: ScoringResult) -> None:
        assert two_path_result.preferred_path.path.name == "alpha"

    def test_alpha_rank_1(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "alpha")
        assert sp.rank == 1

    def test_beta_rank_2(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "beta")
        assert sp.rank == 2

    def test_alpha_throughput_score(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "alpha")
        assert sp.throughput_score == pytest.approx(100.0)

    def test_beta_throughput_score(self, two_path_result: ScoringResult) -> None:
        sp = next(s for s in two_path_result.scored_paths if s.path.name == "beta")
        assert sp.throughput_score == pytest.approx(0.0)


class TestThreePathScoringHandCalc:
    """
    Three paths with distinct metrics. Intermediate path scores a known value.

    Path A: rtt=10ms, loss=0%, jitter=2ms, throughput=100Mbps
    Path B: rtt=30ms, loss=0%, jitter=6ms, throughput=60Mbps
    Path C: rtt=50ms, loss=0%, jitter=10ms, throughput=20Mbps

    All have 0% loss → loss min=max=0 → all loss scores = 50.0

    ── RTT (lower is better) ────────────────────────────────────────
      min=10, max=50
      A: 100×(50-10)/(50-10) = 100.0
      B: 100×(50-30)/(50-10) =  50.0
      C: 100×(50-50)/(50-10) =   0.0

    ── Jitter (lower is better) ─────────────────────────────────────
      min=2, max=10
      A: 100×(10-2)/(10-2)   = 100.0
      B: 100×(10-6)/(10-2)   =  50.0
      C: 100×(10-10)/(10-2)  =   0.0

    ── Throughput (higher is better) ────────────────────────────────
      min=20, max=100
      A: 100×(100-20)/(100-20) = 100.0
      B: 100×( 60-20)/(100-20) =  50.0
      C: 100×( 20-20)/(100-20) =   0.0

    ── Total (weights 0.30/0.30/0.20/0.20) ─────────────────────────
      A: 100×0.30 + 50×0.30 + 100×0.20 + 100×0.20
       = 30 + 15 + 20 + 20 = 85.0

      B: 50×0.30 + 50×0.30 + 50×0.20 + 50×0.20
       = 15 + 15 + 10 + 10 = 50.0

      C: 0×0.30 + 50×0.30 + 0×0.20 + 0×0.20
       = 0 + 15 + 0 + 0 = 15.0
    """

    @pytest.fixture()
    def three_path_result(self) -> ScoringResult:
        pm_a = make_path_metrics(
            "alpha", rtt_mean_ms=10.0, jitter_ms=2.0, loss_pct=0.0, throughput_mbps=100.0
        )
        pm_b = make_path_metrics(
            "beta", rtt_mean_ms=30.0, jitter_ms=6.0, loss_pct=0.0, throughput_mbps=60.0
        )
        pm_c = make_path_metrics(
            "gamma", rtt_mean_ms=50.0, jitter_ms=10.0, loss_pct=0.0, throughput_mbps=20.0
        )
        return score_paths(make_monitor_result(pm_a, pm_b, pm_c))

    def test_alpha_total_score(self, three_path_result: ScoringResult) -> None:
        sp = next(s for s in three_path_result.scored_paths if s.path.name == "alpha")
        assert sp.total_score == pytest.approx(85.0)

    def test_beta_total_score(self, three_path_result: ScoringResult) -> None:
        sp = next(s for s in three_path_result.scored_paths if s.path.name == "beta")
        assert sp.total_score == pytest.approx(50.0)

    def test_gamma_total_score(self, three_path_result: ScoringResult) -> None:
        sp = next(s for s in three_path_result.scored_paths if s.path.name == "gamma")
        assert sp.total_score == pytest.approx(15.0)

    def test_alpha_is_preferred(self, three_path_result: ScoringResult) -> None:
        assert three_path_result.preferred_path.path.name == "alpha"

    def test_ranking_order(self, three_path_result: ScoringResult) -> None:
        names_by_rank = [sp.path.name for sp in three_path_result.scored_paths]
        assert names_by_rank == ["alpha", "beta", "gamma"]

    def test_beta_rtt_score(self, three_path_result: ScoringResult) -> None:
        """B's RTT score = 50.0 (midpoint between A and C)."""
        sp = next(s for s in three_path_result.scored_paths if s.path.name == "beta")
        assert sp.rtt_score == pytest.approx(50.0)

    def test_beta_jitter_score(self, three_path_result: ScoringResult) -> None:
        """B's jitter score = 50.0 (midpoint between A and C)."""
        sp = next(s for s in three_path_result.scored_paths if s.path.name == "beta")
        assert sp.jitter_score == pytest.approx(50.0)

    def test_beta_throughput_score(self, three_path_result: ScoringResult) -> None:
        """B's throughput score = 50.0 (midpoint between A and C)."""
        sp = next(s for s in three_path_result.scored_paths if s.path.name == "beta")
        assert sp.throughput_score == pytest.approx(50.0)


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Availability rule
# ══════════════════════════════════════════════════════════════════════════════


class TestAvailabilityRule:
    """
    A path with 100% probe loss is unavailable.
    A path with any successful probe is available.
    """

    def test_100_percent_loss_is_unavailable(self) -> None:
        pm = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].available is False

    def test_0_percent_loss_is_available(self) -> None:
        pm = make_path_metrics("live", loss_pct=0.0)
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].available is True

    def test_partial_loss_is_available(self) -> None:
        """99% loss is technically reachable — path is available but will score poorly."""
        pm = make_path_metrics("barely", loss_pct=80.0, total_probes=10)
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].available is True

    def test_none_probe_stats_is_unavailable(self) -> None:
        """If probe_stats is None (monitoring crashed), path is unavailable."""
        path = make_path("broken")
        pm = PathMetrics(
            path=path,
            probe_stats=None,
            rtt_stats=None,
            throughput=None,
            error="unexpected error",
        )
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].available is False


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Failed paths cannot be preferred
# ══════════════════════════════════════════════════════════════════════════════


class TestFailedPathsCannotBePreferred:
    """An unavailable path must never be the preferred path."""

    def test_single_unavailable_path_preferred_is_none(self) -> None:
        pm = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm))
        assert result.preferred_path is None

    def test_unavailable_path_not_preferred_when_healthy_exists(self) -> None:
        pm_dead = make_unavailable_path_metrics("dead")
        pm_live = make_path_metrics("live")
        result = score_paths(make_monitor_result(pm_dead, pm_live))
        assert result.preferred_path is not None
        assert result.preferred_path.path.name == "live"

    def test_unavailable_path_ranks_after_available(self) -> None:
        pm_dead = make_unavailable_path_metrics("dead")
        pm_live = make_path_metrics("live")
        result = score_paths(make_monitor_result(pm_dead, pm_live))

        live_sp = next(s for s in result.scored_paths if s.path.name == "live")
        dead_sp = next(s for s in result.scored_paths if s.path.name == "dead")
        assert live_sp.rank < dead_sp.rank

    def test_unavailable_path_all_scores_are_zero(self) -> None:
        pm_dead = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm_dead))
        sp = result.scored_paths[0]
        assert sp.rtt_score        == pytest.approx(0.0)
        assert sp.loss_score       == pytest.approx(0.0)
        assert sp.jitter_score     == pytest.approx(0.0)
        assert sp.throughput_score == pytest.approx(0.0)
        assert sp.total_score      == pytest.approx(0.0)

    def test_unavailable_path_still_appears_in_scored_paths(self) -> None:
        """Even failed paths must appear — the result is inspectable."""
        pm_dead = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm_dead))
        assert len(result.scored_paths) == 1
        assert result.scored_paths[0].path.name == "dead"
        assert result.scored_paths[0].available is False

    def test_two_unavailable_paths_preferred_is_none(self) -> None:
        pm_a = make_unavailable_path_metrics("dead-a")
        pm_b = make_unavailable_path_metrics("dead-b")
        result = score_paths(make_monitor_result(pm_a, pm_b))
        assert result.preferred_path is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: Ranking with multiple paths
# ══════════════════════════════════════════════════════════════════════════════


class TestRanking:
    """Paths must be correctly ranked by total_score descending."""

    def test_higher_score_gets_lower_rank_number(self) -> None:
        """Rank 1 = best (highest score)."""
        pm_good = make_path_metrics("good", rtt_mean_ms=10.0, throughput_mbps=100.0)
        pm_poor = make_path_metrics("poor", rtt_mean_ms=100.0, throughput_mbps=10.0)
        result = score_paths(make_monitor_result(pm_good, pm_poor))

        good_sp = next(s for s in result.scored_paths if s.path.name == "good")
        poor_sp = next(s for s in result.scored_paths if s.path.name == "poor")
        assert good_sp.rank == 1
        assert poor_sp.rank == 2

    def test_ranks_are_contiguous(self) -> None:
        """Ranks must be 1, 2, 3, ... with no gaps."""
        pms = [make_path_metrics(f"p{i}", rtt_mean_ms=float(i * 10 + 10)) for i in range(4)]
        result = score_paths(make_monitor_result(*pms))
        ranks = sorted(sp.rank for sp in result.scored_paths)
        assert ranks == list(range(1, 5))

    def test_all_paths_have_a_rank(self) -> None:
        pms = [make_path_metrics(f"p{i}") for i in range(3)]
        result = score_paths(make_monitor_result(*pms))
        assert all(sp.rank > 0 for sp in result.scored_paths)

    def test_scored_paths_ordered_by_rank(self) -> None:
        """scored_paths list must be sorted by rank ascending."""
        pms = [make_path_metrics(f"p{i}", rtt_mean_ms=float((4 - i) * 10)) for i in range(4)]
        result = score_paths(make_monitor_result(*pms))
        ranks = [sp.rank for sp in result.scored_paths]
        assert ranks == sorted(ranks)


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: Tie handling — deterministic by path name
# ══════════════════════════════════════════════════════════════════════════════


class TestTieHandling:
    """
    When two paths have the same total_score, the tie is broken by path.name
    ascending (lexicographic). This is deterministic and operator-visible.
    """

    def test_identical_metrics_tie_broken_alphabetically(self) -> None:
        """
        Two paths with exactly the same metrics:
          - Both score 50.0 (single path on each metric → max==min → 50.0)
          - Wait — with two identical paths, all metrics have max==min → all 50.0
          - Tie: broken by name → "alpha" < "beta" → alpha is rank 1
        """
        pm_a = make_path_metrics("beta",  rtt_mean_ms=20.0, jitter_ms=2.0, throughput_mbps=50.0)
        pm_b = make_path_metrics("alpha", rtt_mean_ms=20.0, jitter_ms=2.0, throughput_mbps=50.0)
        result = score_paths(make_monitor_result(pm_a, pm_b))

        # Both have max==min for all metrics → both score 50.0
        scores = [sp.total_score for sp in result.scored_paths]
        assert scores[0] == pytest.approx(scores[1])

        # Tie broken alphabetically: "alpha" < "beta"
        assert result.preferred_path.path.name == "alpha"

    def test_tie_broken_by_name_not_insertion_order(self) -> None:
        """
        Even if "zebra" is inserted first, "apple" wins the tie.
        """
        pm_z = make_path_metrics("zebra", rtt_mean_ms=30.0, jitter_ms=5.0, throughput_mbps=50.0)
        pm_a = make_path_metrics("apple", rtt_mean_ms=30.0, jitter_ms=5.0, throughput_mbps=50.0)
        pm_m = make_path_metrics("mango", rtt_mean_ms=30.0, jitter_ms=5.0, throughput_mbps=50.0)
        result = score_paths(make_monitor_result(pm_z, pm_a, pm_m))

        # All identical → all 50.0 → tie → alphabetical
        ranked_names = [sp.path.name for sp in result.scored_paths]
        assert ranked_names == ["apple", "mango", "zebra"]

    def test_tie_result_is_deterministic(self) -> None:
        """
        Calling score_paths twice with the same input must produce the same
        preferred path — no randomness involved.
        """
        pm_a = make_path_metrics("alpha", rtt_mean_ms=20.0)
        pm_b = make_path_metrics("beta",  rtt_mean_ms=20.0)
        mr = make_monitor_result(pm_a, pm_b)

        result1 = score_paths(mr)
        result2 = score_paths(mr)
        assert result1.preferred_path.path.name == result2.preferred_path.path.name


# ══════════════════════════════════════════════════════════════════════════════
# Part 9: Custom weights
# ══════════════════════════════════════════════════════════════════════════════


class TestCustomWeights:
    """Custom weights change the total score and can affect ranking."""

    def test_throughput_heavy_weights_favour_high_throughput_path(self) -> None:
        """
        If throughput weight is 1.0 (and others 0.0), the path with the
        highest throughput wins regardless of RTT or loss.

        Path A: rtt=10ms (best),  throughput=20Mbps  (worst)
        Path B: rtt=100ms (worst), throughput=200Mbps (best)

        With throughput-only weights → B wins.
        """
        pm_a = make_path_metrics("lowrtt",   rtt_mean_ms=10.0,  throughput_mbps=20.0)
        pm_b = make_path_metrics("hightput", rtt_mean_ms=100.0, throughput_mbps=200.0)
        weights = ScoringWeights(rtt=0.0, loss=0.0, jitter=0.0, throughput=1.0)
        result = score_paths(make_monitor_result(pm_a, pm_b), weights=weights)
        assert result.preferred_path.path.name == "hightput"

    def test_rtt_only_weights_favour_low_rtt_path(self) -> None:
        pm_a = make_path_metrics("lowrtt",   rtt_mean_ms=5.0,   throughput_mbps=10.0)
        pm_b = make_path_metrics("highrtt",  rtt_mean_ms=200.0, throughput_mbps=500.0)
        weights = ScoringWeights(rtt=1.0, loss=0.0, jitter=0.0, throughput=0.0)
        result = score_paths(make_monitor_result(pm_a, pm_b), weights=weights)
        assert result.preferred_path.path.name == "lowrtt"

    def test_custom_weights_stored_in_result(self) -> None:
        pm = make_path_metrics("x")
        w = ScoringWeights(rtt=0.50, loss=0.25, jitter=0.15, throughput=0.10)
        result = score_paths(make_monitor_result(pm), weights=w)
        assert result.weights is w

    def test_default_weights_used_when_none_passed(self) -> None:
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm), weights=None)
        assert result.weights.rtt        == pytest.approx(0.30)
        assert result.weights.loss       == pytest.approx(0.30)
        assert result.weights.jitter     == pytest.approx(0.20)
        assert result.weights.throughput == pytest.approx(0.20)

    def test_custom_weights_change_total_score(self) -> None:
        """
        Path A: rtt=10ms (best), throughput=10Mbps (worst)
        Path B: rtt=100ms (worst), throughput=100Mbps (best)

        With equal weights (0.25 each, loss/jitter tied at 50):
          A: rtt=100, loss=50, jitter=50, tput=0  → 100×0.25+50×0.25+50×0.25+0×0.25 = 50.0
          B: rtt=0,   loss=50, jitter=50, tput=100 → 0×0.25+50×0.25+50×0.25+100×0.25 = 50.0
          → tie, broken by name "alpha" < "beta" (if named so)

        With throughput-only weight, B wins.
        This demonstrates weights change outcomes.
        """
        pm_a = make_path_metrics("alpha", rtt_mean_ms=10.0, throughput_mbps=10.0)
        pm_b = make_path_metrics("beta",  rtt_mean_ms=100.0, throughput_mbps=100.0)

        # throughput-only: B wins
        w_tput = ScoringWeights(rtt=0.0, loss=0.0, jitter=0.0, throughput=1.0)
        r_tput = score_paths(make_monitor_result(pm_a, pm_b), weights=w_tput)
        assert r_tput.preferred_path.path.name == "beta"

        # rtt-only: A wins
        w_rtt = ScoringWeights(rtt=1.0, loss=0.0, jitter=0.0, throughput=0.0)
        r_rtt = score_paths(make_monitor_result(pm_a, pm_b), weights=w_rtt)
        assert r_rtt.preferred_path.path.name == "alpha"


# ══════════════════════════════════════════════════════════════════════════════
# Part 10: Missing metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestMissingMetrics:
    """
    When a metric is missing (throughput failed, or no RTT data), the path
    receives 0.0 for that metric. The path is not excluded from ranking.
    """

    def test_missing_throughput_gives_zero_throughput_score(self) -> None:
        pm = make_path_metrics(
            "x", throughput_mbps=None, throughput_status=ProbeStatus.REFUSED
        )
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].throughput_score == pytest.approx(0.0)

    def test_missing_throughput_path_still_preferred_if_only_path(self) -> None:
        """A path with missing throughput is still available (probes succeed)."""
        pm = make_path_metrics(
            "x", throughput_mbps=None, throughput_status=ProbeStatus.REFUSED
        )
        result = score_paths(make_monitor_result(pm))
        assert result.preferred_path is not None

    def test_missing_throughput_on_both_paths_both_score_zero_on_throughput(self) -> None:
        """If both paths have no throughput data, both score 0.0 on throughput."""
        pm_a = make_path_metrics("a", throughput_mbps=None, throughput_status=ProbeStatus.REFUSED)
        pm_b = make_path_metrics("b", throughput_mbps=None, throughput_status=ProbeStatus.REFUSED)
        result = score_paths(make_monitor_result(pm_a, pm_b))
        for sp in result.scored_paths:
            assert sp.throughput_score == pytest.approx(0.0)

    def test_path_with_all_metrics_beats_path_with_missing_throughput(self) -> None:
        """
        Path A: complete metrics, 0% loss, low RTT
        Path B: throughput missing (gets 0.0 on throughput score)

        A should score higher and be preferred.
        """
        pm_a = make_path_metrics("complete",  rtt_mean_ms=10.0, throughput_mbps=100.0)
        pm_b = make_path_metrics("nothrput", rtt_mean_ms=10.0,
                                  throughput_mbps=None, throughput_status=ProbeStatus.REFUSED)
        result = score_paths(make_monitor_result(pm_a, pm_b))
        assert result.preferred_path.path.name == "complete"


# ══════════════════════════════════════════════════════════════════════════════
# Part 11: Empty path input
# ══════════════════════════════════════════════════════════════════════════════


class TestEmptyInput:
    """An empty MonitorResult must produce an empty ScoringResult cleanly."""

    def test_empty_monitor_result_returns_scoring_result(self) -> None:
        result = score_paths(MonitorResult(metrics={}))
        assert isinstance(result, ScoringResult)

    def test_empty_monitor_result_has_no_scored_paths(self) -> None:
        result = score_paths(MonitorResult(metrics={}))
        assert result.scored_paths == []

    def test_empty_monitor_result_preferred_is_none(self) -> None:
        result = score_paths(MonitorResult(metrics={}))
        assert result.preferred_path is None

    def test_empty_monitor_result_weights_recorded(self) -> None:
        result = score_paths(MonitorResult(metrics={}))
        assert isinstance(result.weights, ScoringWeights)


# ══════════════════════════════════════════════════════════════════════════════
# Part 12: Inspectable result structure
# ══════════════════════════════════════════════════════════════════════════════


class TestInspectableResult:
    """The result must expose all intermediate scores, not just the winner."""

    def test_scored_path_contains_all_four_component_scores(self) -> None:
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm))
        sp = result.scored_paths[0]
        # All four component scores must be accessible attributes
        assert hasattr(sp, "rtt_score")
        assert hasattr(sp, "loss_score")
        assert hasattr(sp, "jitter_score")
        assert hasattr(sp, "throughput_score")
        assert hasattr(sp, "total_score")

    def test_scored_path_contains_original_metrics(self) -> None:
        """ScoredPath.metrics must be the original PathMetrics, not a copy."""
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].metrics is pm

    def test_scored_path_contains_path_object(self) -> None:
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm))
        assert result.scored_paths[0].path == pm.path

    def test_scoring_result_contains_weights(self) -> None:
        pm = make_path_metrics("x")
        w = ScoringWeights()
        result = score_paths(make_monitor_result(pm), weights=w)
        assert result.weights is w

    def test_scoring_result_contains_all_paths_including_failed(self) -> None:
        pm_live = make_path_metrics("live")
        pm_dead = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm_live, pm_dead))
        names = {sp.path.name for sp in result.scored_paths}
        assert "live" in names
        assert "dead" in names

    def test_str_does_not_raise(self) -> None:
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm))
        s = str(result)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_scored_path_str_does_not_raise(self) -> None:
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm))
        s = str(result.scored_paths[0])
        assert isinstance(s, str)

    def test_result_is_frozen_immutable(self) -> None:
        pm = make_path_metrics("x")
        result = score_paths(make_monitor_result(pm))
        with pytest.raises(Exception):
            result.preferred_path = None  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# Part 13: All-equal metric scores (max == min edge case)
# ══════════════════════════════════════════════════════════════════════════════


class TestEqualMetricEdgeCase:
    """When all paths have the same value on a metric, all score 50.0 on it."""

    def test_all_same_rtt_scores_midpoint(self) -> None:
        pm_a = make_path_metrics("a", rtt_mean_ms=20.0, throughput_mbps=50.0)
        pm_b = make_path_metrics("b", rtt_mean_ms=20.0, throughput_mbps=100.0)
        result = score_paths(make_monitor_result(pm_a, pm_b))
        for sp in result.scored_paths:
            assert sp.rtt_score == pytest.approx(_EQUAL_METRIC_SCORE)

    def test_all_same_loss_scores_midpoint(self) -> None:
        pm_a = make_path_metrics("a", loss_pct=0.0, rtt_mean_ms=10.0)
        pm_b = make_path_metrics("b", loss_pct=0.0, rtt_mean_ms=50.0)
        result = score_paths(make_monitor_result(pm_a, pm_b))
        for sp in result.scored_paths:
            assert sp.loss_score == pytest.approx(_EQUAL_METRIC_SCORE)

    def test_all_same_throughput_scores_midpoint(self) -> None:
        pm_a = make_path_metrics("a", rtt_mean_ms=10.0, throughput_mbps=100.0)
        pm_b = make_path_metrics("b", rtt_mean_ms=50.0, throughput_mbps=100.0)
        result = score_paths(make_monitor_result(pm_a, pm_b))
        for sp in result.scored_paths:
            assert sp.throughput_score == pytest.approx(_EQUAL_METRIC_SCORE)


# ══════════════════════════════════════════════════════════════════════════════
# Part 14: Mixed available / unavailable with hand-calculated scores
# ══════════════════════════════════════════════════════════════════════════════


class TestMixedAvailability:
    """
    Mix of one available and one unavailable path.
    Normalization range is computed from available paths only.
    Unavailable path scores 0.0 on all metrics regardless.
    """

    def test_unavailable_path_excluded_from_normalization_range(self) -> None:
        """
        With one available path, all metrics have max==min → scores = 50.0.
        The unavailable path does NOT pull the range (its 100% loss should
        not set loss_max=100, which would make the available path's 0% loss
        score non-100).
        """
        pm_live = make_path_metrics("live", loss_pct=0.0)
        pm_dead = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm_live, pm_dead))

        live_sp = next(s for s in result.scored_paths if s.path.name == "live")
        # Only one available path → all max==min → all 50.0
        assert live_sp.loss_score == pytest.approx(_EQUAL_METRIC_SCORE)

    def test_two_available_paths_scored_correctly_ignoring_unavailable(self) -> None:
        """
        Two available paths + one dead path.
        Dead path must not affect normalization range of the two live paths.

        Live A: rtt=10ms, Live B: rtt=50ms
        RTT normalization (live only): min=10, max=50
          A: 100×(50-10)/(50-10) = 100.0
          B: 100×(50-50)/(50-10) = 0.0
        """
        pm_a    = make_path_metrics("alpha", rtt_mean_ms=10.0)
        pm_b    = make_path_metrics("beta",  rtt_mean_ms=50.0)
        pm_dead = make_unavailable_path_metrics("dead")
        result = score_paths(make_monitor_result(pm_a, pm_b, pm_dead))

        sp_a = next(s for s in result.scored_paths if s.path.name == "alpha")
        sp_b = next(s for s in result.scored_paths if s.path.name == "beta")
        assert sp_a.rtt_score == pytest.approx(100.0)
        assert sp_b.rtt_score == pytest.approx(0.0)
