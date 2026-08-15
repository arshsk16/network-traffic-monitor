"""
Tests for core/stats.py — RTT statistics and jitter estimation.

Test strategy
─────────────
All inputs are manually constructed lists/tuples of RTT values.
No network I/O. No timing assertions. Pure deterministic math.

Because the statistics are calculated from arbitrary input floats, every
assertion is exact (==) or uses pytest.approx() only where floating-point
arithmetic demands it (division-based averages).

The jitter calculation works on these concrete examples:

  samples = [20.0, 22.0, 21.0, 30.0]
  consecutive diffs: |22-20|=2, |21-22|=1, |30-21|=9
  jitter = (2 + 1 + 9) / 3 = 4.0ms  (exactly 4.0 — no float rounding)

  samples = [10.0, 20.0, 10.0, 20.0]
  consecutive diffs: 10, 10, 10
  jitter = 30 / 3 = 10.0ms

  samples = [5.0, 5.0, 5.0]
  consecutive diffs: 0, 0
  jitter = 0.0ms  (stable path)
"""

import pytest

from core.stats import RttStats, compute_rtt_stats


# ── Tests: empty sample set ───────────────────────────────────────────────────


class TestEmptySamples:
    """With no RTT samples, all statistics are undefined (None)."""

    def test_returns_rtt_stats(self) -> None:
        result = compute_rtt_stats([])
        assert isinstance(result, RttStats)

    def test_count_is_zero(self) -> None:
        result = compute_rtt_stats([])
        assert result.count == 0

    def test_mean_is_none(self) -> None:
        result = compute_rtt_stats([])
        assert result.mean_ms is None

    def test_min_is_none(self) -> None:
        result = compute_rtt_stats([])
        assert result.min_ms is None

    def test_max_is_none(self) -> None:
        result = compute_rtt_stats([])
        assert result.max_ms is None

    def test_jitter_is_none(self) -> None:
        """
        With 0 samples, jitter is None (undefined) — not 0.0.
        Returning 0.0 would falsely imply a stable measurement was taken.
        """
        result = compute_rtt_stats([])
        assert result.jitter_ms is None

    def test_accepts_empty_tuple(self) -> None:
        """Should accept tuple input as well as list."""
        result = compute_rtt_stats(())
        assert result.count == 0

    def test_accepts_empty_list(self) -> None:
        result = compute_rtt_stats([])
        assert result.count == 0


# ── Tests: single sample ──────────────────────────────────────────────────────


class TestSingleSample:
    """
    With exactly one RTT sample, mean/min/max are defined.
    Jitter is 0.0 (not None): zero variation was observed.

    There are no consecutive pairs with one sample, so there are no
    differences to average. We define this as 0.0 rather than None because
    "we measured once and saw no variation" is a meaningful statement.
    """

    def test_count_is_one(self) -> None:
        result = compute_rtt_stats([15.0])
        assert result.count == 1

    def test_mean_equals_sample(self) -> None:
        result = compute_rtt_stats([15.0])
        assert result.mean_ms == 15.0

    def test_min_equals_sample(self) -> None:
        result = compute_rtt_stats([15.0])
        assert result.min_ms == 15.0

    def test_max_equals_sample(self) -> None:
        result = compute_rtt_stats([15.0])
        assert result.max_ms == 15.0

    def test_jitter_is_zero(self) -> None:
        """One sample → no consecutive pairs → 0.0 variation observed."""
        result = compute_rtt_stats([15.0])
        assert result.jitter_ms == 0.0

    def test_jitter_is_not_none(self) -> None:
        """Single sample must give 0.0, not None — it's a defined value."""
        result = compute_rtt_stats([42.0])
        assert result.jitter_ms is not None


# ── Tests: stable RTT samples → zero jitter ───────────────────────────────────


class TestStableSamples:
    """When all samples are identical, jitter must be exactly 0.0."""

    def test_all_same_values_jitter_is_zero(self) -> None:
        result = compute_rtt_stats([20.0, 20.0, 20.0, 20.0, 20.0])
        assert result.jitter_ms == 0.0

    def test_all_same_mean_equals_value(self) -> None:
        result = compute_rtt_stats([10.0, 10.0, 10.0])
        assert result.mean_ms == 10.0

    def test_all_same_min_equals_max(self) -> None:
        result = compute_rtt_stats([10.0, 10.0, 10.0])
        assert result.min_ms == result.max_ms == 10.0

    def test_two_identical_samples(self) -> None:
        """Minimum case for computing a consecutive difference: 2 samples."""
        result = compute_rtt_stats([5.0, 5.0])
        assert result.jitter_ms == 0.0
        assert result.count == 2


# ── Tests: known sequence → exact expected jitter ─────────────────────────────


class TestKnownJitterCalculation:
    """
    Verify the jitter formula against manually computed expected values.

    Manual walkthrough for [20.0, 22.0, 21.0, 30.0]:
      consecutive diffs: |22-20|=2, |21-22|=1, |30-21|=9
      mean of diffs:     (2 + 1 + 9) / 3 = 4.0ms
    """

    def test_four_sample_sequence_jitter(self) -> None:
        samples = [20.0, 22.0, 21.0, 30.0]
        result = compute_rtt_stats(samples)
        assert result.jitter_ms == pytest.approx(4.0)

    def test_four_sample_sequence_mean(self) -> None:
        samples = [20.0, 22.0, 21.0, 30.0]
        result = compute_rtt_stats(samples)
        # (20 + 22 + 21 + 30) / 4 = 93 / 4 = 23.25
        assert result.mean_ms == pytest.approx(23.25)

    def test_four_sample_sequence_min(self) -> None:
        samples = [20.0, 22.0, 21.0, 30.0]
        result = compute_rtt_stats(samples)
        assert result.min_ms == 20.0

    def test_four_sample_sequence_max(self) -> None:
        samples = [20.0, 22.0, 21.0, 30.0]
        result = compute_rtt_stats(samples)
        assert result.max_ms == 30.0

    def test_alternating_sequence_jitter(self) -> None:
        """
        [10.0, 20.0, 10.0, 20.0]
        diffs: |20-10|=10, |10-20|=10, |20-10|=10
        jitter: (10+10+10)/3 = 10.0ms
        """
        samples = [10.0, 20.0, 10.0, 20.0]
        result = compute_rtt_stats(samples)
        assert result.jitter_ms == pytest.approx(10.0)

    def test_two_sample_jitter(self) -> None:
        """
        [5.0, 15.0]
        diffs: |15-5| = 10
        jitter: 10 / 1 = 10.0ms
        """
        samples = [5.0, 15.0]
        result = compute_rtt_stats(samples)
        assert result.jitter_ms == pytest.approx(10.0)

    def test_three_sample_jitter(self) -> None:
        """
        [100.0, 110.0, 90.0]
        diffs: |110-100|=10, |90-110|=20
        jitter: (10+20)/2 = 15.0ms
        """
        samples = [100.0, 110.0, 90.0]
        result = compute_rtt_stats(samples)
        assert result.jitter_ms == pytest.approx(15.0)

    def test_count_matches_input_length(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_rtt_stats(samples)
        assert result.count == 5


# ── Tests: highly variable samples → larger jitter ───────────────────────────


class TestHighVariability:
    """High variation in samples should produce a large jitter value."""

    def test_high_variance_jitter_larger_than_low_variance(self) -> None:
        """
        Stable path: [10.0, 11.0, 10.0, 11.0]
          diffs: 1, 1, 1 → jitter = 1.0ms
        Unstable path: [5.0, 50.0, 8.0, 45.0]
          diffs: 45, 42, 37 → jitter = (45+42+37)/3 = 41.33ms
        """
        stable = compute_rtt_stats([10.0, 11.0, 10.0, 11.0])
        unstable = compute_rtt_stats([5.0, 50.0, 8.0, 45.0])
        assert unstable.jitter_ms > stable.jitter_ms

    def test_stable_path_jitter_near_zero(self) -> None:
        stable = compute_rtt_stats([10.0, 11.0, 10.0, 11.0])
        assert stable.jitter_ms == pytest.approx(1.0)

    def test_unstable_path_jitter_value(self) -> None:
        # [5.0, 50.0, 8.0, 45.0]
        # diffs: 45, 42, 37 → (45+42+37)/3 = 124/3 ≈ 41.333
        unstable = compute_rtt_stats([5.0, 50.0, 8.0, 45.0])
        assert unstable.jitter_ms == pytest.approx(124.0 / 3, rel=1e-6)


# ── Tests: mean, min, max calculations ───────────────────────────────────────


class TestMeanMinMax:
    """Verify mean, min, and max against known inputs."""

    def test_mean_of_simple_values(self) -> None:
        result = compute_rtt_stats([10.0, 20.0, 30.0])
        assert result.mean_ms == pytest.approx(20.0)

    def test_min_is_smallest_sample(self) -> None:
        result = compute_rtt_stats([30.0, 5.0, 20.0, 15.0])
        assert result.min_ms == 5.0

    def test_max_is_largest_sample(self) -> None:
        result = compute_rtt_stats([30.0, 5.0, 20.0, 15.0])
        assert result.max_ms == 30.0

    def test_mean_single_decimal_sample(self) -> None:
        result = compute_rtt_stats([1.5, 2.5, 3.5, 4.5])
        # (1.5 + 2.5 + 3.5 + 4.5) / 4 = 12 / 4 = 3.0
        assert result.mean_ms == pytest.approx(3.0)

    def test_min_max_on_single_element(self) -> None:
        result = compute_rtt_stats([42.0])
        assert result.min_ms == 42.0
        assert result.max_ms == 42.0

    def test_min_not_equal_max_when_varied(self) -> None:
        result = compute_rtt_stats([10.0, 20.0])
        assert result.min_ms != result.max_ms


# ── Tests: input accepts both list and tuple ──────────────────────────────────


class TestInputTypes:
    """compute_rtt_stats() should accept both list and tuple inputs."""

    def test_accepts_list(self) -> None:
        result = compute_rtt_stats([10.0, 20.0])
        assert result.count == 2

    def test_accepts_tuple(self) -> None:
        """rtt_values_ms in ProbeStats is a tuple — must work directly."""
        result = compute_rtt_stats((10.0, 20.0))
        assert result.count == 2

    def test_list_and_tuple_same_result(self) -> None:
        samples_list = [15.0, 25.0, 20.0]
        samples_tuple = (15.0, 25.0, 20.0)
        r1 = compute_rtt_stats(samples_list)
        r2 = compute_rtt_stats(samples_tuple)
        assert r1.jitter_ms == r2.jitter_ms
        assert r1.mean_ms == r2.mean_ms


# ── Tests: invalid values ─────────────────────────────────────────────────────


class TestInvalidValues:
    """Negative RTT values are physically impossible and must be rejected."""

    def test_negative_sample_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            compute_rtt_stats([-1.0, 20.0])

    def test_single_negative_sample_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_rtt_stats([-5.0])

    def test_zero_is_valid(self) -> None:
        """
        0.0ms is technically impossible on a real network but not a
        programmer error — it could be a test fixture artifact.
        We accept it rather than over-validating.
        """
        result = compute_rtt_stats([0.0, 1.0])
        assert result.count == 2

    def test_positive_float_is_valid(self) -> None:
        result = compute_rtt_stats([0.001, 0.5, 1000.0])
        assert result.count == 3


# ── Tests: RttStats is immutable ──────────────────────────────────────────────


class TestImmutability:
    """RttStats must be immutable — it is a frozen dataclass."""

    def test_cannot_set_attribute(self) -> None:
        result = compute_rtt_stats([10.0, 20.0])
        with pytest.raises((AttributeError, TypeError)):
            result.mean_ms = 99.0  # type: ignore[misc]

    def test_result_is_rtt_stats_instance(self) -> None:
        result = compute_rtt_stats([10.0])
        assert isinstance(result, RttStats)
