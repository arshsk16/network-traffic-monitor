"""
tests/test_failover.py — Deterministic unit tests for core/failover.py.

Test strategy
─────────────
All tests construct ScoredPath / ScoringResult objects directly — no real
network calls, no monitoring, no scoring engine calls. This makes every test
fast, deterministic, and independent of external state.

We need to build ScoredPath and ScoringResult objects manually. ScoredPath
requires a PathMetrics, which in turn requires a ProbeStats. Rather than
importing the full chain, small factory helpers construct the minimum valid
objects needed to satisfy the types. The scoring values (rtt_score, etc.)
are irrelevant for failover logic — only path.name and available matter.

What is tested
──────────────
Part 1:  Initial state (no preferred path yet)
Part 2:  INITIAL_SELECTION — None → A
Part 3:  NO_CHANGE — A → A
Part 4:  FAILOVER — A → B
Part 5:  NO_AVAILABLE_PATH — A → None and None → None
Part 6:  Unavailable paths never become preferred
Part 7:  State persists across multiple update() calls
Part 8:  Transition result contents (paths, event_type)
Part 9:  PathTransition convenience helpers (is_failover, is_change)
Part 10: reset() returns state to initial condition
Part 11: PathState repr and PathTransition str do not raise
"""

from __future__ import annotations

import pytest

from core.failover import PathState, PathTransition, TransitionType
from core.monitor import PathMetrics
from core.path import Path
from core.scoring import ScoredPath, ScoringResult, ScoringWeights

# We need the minimum imports to build PathMetrics objects
from core.loss import ProbeStats
from core.stats import RttStats
from core.throughput import ThroughputResult
from core.probe import ProbeStatus


# ── Builder helpers ────────────────────────────────────────────────────────────
# All test data is constructed here rather than using real monitoring/scoring
# code. This isolates failover tests from scoring bugs.


def _make_path(name: str) -> Path:
    """Create a minimal valid Path."""
    return Path(name=name, host="127.0.0.1", port=9000)


def _make_minimal_metrics(name: str) -> PathMetrics:
    """
    Build the minimum valid PathMetrics for an available path.

    probe_stats.loss_rate = 0.0 → path will be available.
    rtt_stats, throughput have placeholder values — not used by failover logic.
    """
    path = _make_path(name)
    probe_stats = ProbeStats(
        host=path.host,
        port=path.port,
        total=1,
        successful=1,
        failed=0,
        loss_rate=0.0,
        rtt_values_ms=(10.0,),
        raw_results=(),
    )
    rtt_stats = RttStats(count=1, mean_ms=10.0, min_ms=10.0, max_ms=10.0, jitter_ms=0.0)
    throughput = ThroughputResult(
        host=path.host,
        port=path.port,
        status=ProbeStatus.SUCCESS,
        bytes_transferred=65536,
        elapsed_seconds=0.01,
        throughput_bps=65536 * 8 / 0.01,
        throughput_mbps=(65536 * 8 / 0.01) / 1_000_000,
        error_message=None,
    )
    return PathMetrics(
        path=path,
        probe_stats=probe_stats,
        rtt_stats=rtt_stats,
        throughput=throughput,
    )


def _make_unavailable_metrics(name: str) -> PathMetrics:
    """
    Build PathMetrics for a fully-unavailable path (100% probe loss).

    probe_stats.loss_rate = 100.0 → scoring engine marks this unavailable.
    """
    path = _make_path(name)
    probe_stats = ProbeStats(
        host=path.host,
        port=path.port,
        total=5,
        successful=0,
        failed=5,
        loss_rate=100.0,
        rtt_values_ms=(),
        raw_results=(),
    )
    rtt_stats = RttStats(count=0, mean_ms=None, min_ms=None, max_ms=None, jitter_ms=None)
    throughput = ThroughputResult(
        host=path.host,
        port=path.port,
        status=ProbeStatus.REFUSED,
        bytes_transferred=None,
        elapsed_seconds=None,
        throughput_bps=None,
        throughput_mbps=None,
        error_message="connection refused",
    )
    return PathMetrics(
        path=path,
        probe_stats=probe_stats,
        rtt_stats=rtt_stats,
        throughput=throughput,
    )


_DEFAULT_WEIGHTS = ScoringWeights()


def _make_scored_path(
    name: str,
    available: bool = True,
    total_score: float = 75.0,
    rank: int = 1,
) -> ScoredPath:
    """Build a ScoredPath with controlled availability and score values."""
    metrics = (
        _make_minimal_metrics(name)
        if available
        else _make_unavailable_metrics(name)
    )
    return ScoredPath(
        path=metrics.path,
        metrics=metrics,
        available=available,
        rtt_score=total_score,
        loss_score=total_score,
        jitter_score=total_score,
        throughput_score=total_score,
        total_score=total_score,
        rank=rank,
    )


def _make_scoring_result(
    preferred: ScoredPath | None,
    all_paths: list[ScoredPath] | None = None,
) -> ScoringResult:
    """
    Build a ScoringResult where preferred_path is the given ScoredPath.

    If all_paths is not provided, it defaults to [preferred] if preferred
    is not None, else [].
    """
    if all_paths is None:
        all_paths = [preferred] if preferred is not None else []
    return ScoringResult(
        scored_paths=all_paths,
        preferred_path=preferred,
        weights=_DEFAULT_WEIGHTS,
    )


def _scoring_result_with_preferred(name: str, score: float = 75.0) -> ScoringResult:
    """Convenience: one available preferred path by name."""
    sp = _make_scored_path(name, available=True, total_score=score, rank=1)
    return _make_scoring_result(preferred=sp)


def _scoring_result_no_preferred() -> ScoringResult:
    """Convenience: no preferred path (all unavailable or no paths)."""
    return _make_scoring_result(preferred=None, all_paths=[])


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Initial state
# ══════════════════════════════════════════════════════════════════════════════


class TestInitialState:
    """A freshly-created PathState has no preferred path."""

    def test_initial_preferred_path_is_none(self) -> None:
        state = PathState()
        assert state.current_preferred_path is None

    def test_initial_scored_path_is_none(self) -> None:
        state = PathState()
        assert state.current_scored_path is None

    def test_initial_has_preferred_path_is_false(self) -> None:
        state = PathState()
        assert state.has_preferred_path is False

    def test_repr_shows_none(self) -> None:
        state = PathState()
        r = repr(state)
        assert "None" in r


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: INITIAL_SELECTION — None → A
# ══════════════════════════════════════════════════════════════════════════════


class TestInitialSelection:
    """First update that produces an available path → INITIAL_SELECTION."""

    @pytest.fixture()
    def state_and_transition(self) -> tuple[PathState, PathTransition]:
        state = PathState()
        result = _scoring_result_with_preferred("primary")
        transition = state.update(result)
        return state, transition

    def test_event_type_is_initial_selection(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.event_type == TransitionType.INITIAL_SELECTION

    def test_previous_path_is_none(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.previous_path is None

    def test_new_path_is_selected_path(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.new_path is not None
        assert transition.new_path.name == "primary"

    def test_state_updated_to_selected_path(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        state, _ = state_and_transition
        assert state.current_preferred_path is not None
        assert state.current_preferred_path.name == "primary"

    def test_has_preferred_path_is_true_after_selection(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        state, _ = state_and_transition
        assert state.has_preferred_path is True

    def test_initial_selection_is_not_failover(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        """INITIAL_SELECTION must not be classified as FAILOVER."""
        _, transition = state_and_transition
        assert transition.event_type != TransitionType.FAILOVER
        assert transition.is_failover() is False

    def test_initial_selection_is_a_change(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        """is_change() should return True for INITIAL_SELECTION."""
        _, transition = state_and_transition
        assert transition.is_change() is True


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: NO_CHANGE — A → A
# ══════════════════════════════════════════════════════════════════════════════


class TestNoChange:
    """Same preferred path across two consecutive updates → NO_CHANGE."""

    @pytest.fixture()
    def state_and_transition(self) -> tuple[PathState, PathTransition]:
        state = PathState()
        # First update: select "primary"
        state.update(_scoring_result_with_preferred("primary"))
        # Second update: "primary" still preferred
        transition = state.update(_scoring_result_with_preferred("primary"))
        return state, transition

    def test_event_type_is_no_change(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.event_type == TransitionType.NO_CHANGE

    def test_previous_path_is_primary(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.previous_path is not None
        assert transition.previous_path.name == "primary"

    def test_new_path_is_primary(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.new_path is not None
        assert transition.new_path.name == "primary"

    def test_state_still_primary(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        state, _ = state_and_transition
        assert state.current_preferred_path.name == "primary"

    def test_no_change_is_not_failover(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.is_failover() is False

    def test_no_change_is_not_a_change(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        """is_change() returns False for NO_CHANGE."""
        _, transition = state_and_transition
        assert transition.is_change() is False

    def test_repeated_same_path_never_produces_failover(self) -> None:
        """10 consecutive updates with the same path → never FAILOVER."""
        state = PathState()
        for _ in range(10):
            t = state.update(_scoring_result_with_preferred("primary"))
        # Last transition must be NO_CHANGE
        assert t.event_type == TransitionType.NO_CHANGE


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: FAILOVER — A → B
# ══════════════════════════════════════════════════════════════════════════════


class TestFailover:
    """Preferred path changes from A to B → FAILOVER."""

    @pytest.fixture()
    def state_and_transition(self) -> tuple[PathState, PathTransition]:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_with_preferred("backup"))
        return state, transition

    def test_event_type_is_failover(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.event_type == TransitionType.FAILOVER

    def test_previous_path_is_primary(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.previous_path is not None
        assert transition.previous_path.name == "primary"

    def test_new_path_is_backup(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.new_path is not None
        assert transition.new_path.name == "backup"

    def test_state_updated_to_backup(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        state, _ = state_and_transition
        assert state.current_preferred_path.name == "backup"

    def test_failover_is_failover(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.is_failover() is True

    def test_failover_is_a_change(
        self, state_and_transition: tuple[PathState, PathTransition]
    ) -> None:
        _, transition = state_and_transition
        assert transition.is_change() is True

    def test_failover_due_to_unavailability(self) -> None:
        """
        Primary was preferred. Primary becomes unavailable. Backup is available.
        Expected: FAILOVER primary → backup.
        """
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))

        # New cycle: primary is unavailable, backup is the new preferred
        dead_primary = _make_scored_path("primary", available=False, total_score=0.0, rank=2)
        live_backup = _make_scored_path("backup", available=True, total_score=80.0, rank=1)
        new_result = _make_scoring_result(
            preferred=live_backup,
            all_paths=[live_backup, dead_primary],
        )
        transition = state.update(new_result)

        assert transition.event_type == TransitionType.FAILOVER
        assert transition.previous_path.name == "primary"
        assert transition.new_path.name == "backup"

    def test_previous_scored_path_in_failover_result(self) -> None:
        """
        When the previous path still appears in the new result (even if
        demoted), previous_scored_path should be populated.
        """
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))

        # New result: backup scores higher, primary still present but rank 2
        sp_primary = _make_scored_path("primary", available=True, total_score=60.0, rank=2)
        sp_backup  = _make_scored_path("backup",  available=True, total_score=90.0, rank=1)
        new_result = _make_scoring_result(
            preferred=sp_backup,
            all_paths=[sp_backup, sp_primary],
        )
        transition = state.update(new_result)

        assert transition.event_type == TransitionType.FAILOVER
        assert transition.previous_scored_path is not None
        assert transition.previous_scored_path.path.name == "primary"


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: NO_AVAILABLE_PATH
# ══════════════════════════════════════════════════════════════════════════════


class TestNoAvailablePath:
    """All paths unavailable → NO_AVAILABLE_PATH."""

    def test_a_to_none_is_no_available_path(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        assert transition.event_type == TransitionType.NO_AVAILABLE_PATH

    def test_none_to_none_is_no_available_path(self) -> None:
        """State starts at None; still no paths available → NO_AVAILABLE_PATH."""
        state = PathState()
        transition = state.update(_scoring_result_no_preferred())
        assert transition.event_type == TransitionType.NO_AVAILABLE_PATH

    def test_new_path_is_none_on_no_available(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        assert transition.new_path is None

    def test_previous_path_is_primary_on_a_to_none(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        assert transition.previous_path is not None
        assert transition.previous_path.name == "primary"

    def test_state_preferred_becomes_none(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        state.update(_scoring_result_no_preferred())
        assert state.current_preferred_path is None
        assert state.has_preferred_path is False

    def test_no_available_path_is_a_change(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        assert transition.is_change() is True

    def test_no_available_path_is_not_failover(self) -> None:
        """NO_AVAILABLE_PATH is distinct from FAILOVER."""
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        assert transition.is_failover() is False

    def test_all_paths_unavailable_preferred_is_none(self) -> None:
        """Multiple unavailable paths → preferred is still None."""
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))

        dead_a = _make_scored_path("primary", available=False, total_score=0.0, rank=1)
        dead_b = _make_scored_path("backup",  available=False, total_score=0.0, rank=2)
        result = _make_scoring_result(preferred=None, all_paths=[dead_a, dead_b])

        state.update(result)
        assert state.current_preferred_path is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: Unavailable paths never become preferred
# ══════════════════════════════════════════════════════════════════════════════


class TestUnavailablePathsNeverPreferred:
    """
    The scoring engine already ensures preferred_path is always available.
    These tests confirm that state transitions reflect this correctly.
    """

    def test_unavailable_path_not_in_preferred(self) -> None:
        """
        If scoring returns a result where preferred_path is an available path,
        an unavailable sibling must not become current_preferred_path.
        """
        state = PathState()
        dead = _make_scored_path("dead",  available=False, total_score=0.0, rank=2)
        live = _make_scored_path("live",  available=True,  total_score=80.0, rank=1)
        result = _make_scoring_result(preferred=live, all_paths=[live, dead])
        state.update(result)
        assert state.current_preferred_path.name == "live"

    def test_after_failover_to_unavailable_no_preferred(self) -> None:
        """
        If the scoring engine returns preferred_path=None (because the only
        path is unavailable), state must also be None.
        """
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))

        dead = _make_scored_path("primary", available=False, total_score=0.0, rank=1)
        result = _make_scoring_result(preferred=None, all_paths=[dead])
        state.update(result)

        assert state.current_preferred_path is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 7: State persists across multiple update() calls
# ══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    """State must correctly accumulate across multiple update() calls."""

    def test_full_lifecycle(self) -> None:
        """
        Simulate a realistic monitoring sequence:
          1. Boot: None → primary (INITIAL_SELECTION)
          2. Healthy: primary → primary (NO_CHANGE)
          3. Healthy: primary → primary (NO_CHANGE)
          4. Primary fails: primary → backup (FAILOVER)
          5. Backup healthy: backup → backup (NO_CHANGE)
          6. All fail: backup → None (NO_AVAILABLE_PATH)
          7. Primary recovers: None → primary (INITIAL_SELECTION)
        """
        state = PathState()

        t1 = state.update(_scoring_result_with_preferred("primary"))
        assert t1.event_type == TransitionType.INITIAL_SELECTION
        assert state.current_preferred_path.name == "primary"

        t2 = state.update(_scoring_result_with_preferred("primary"))
        assert t2.event_type == TransitionType.NO_CHANGE

        t3 = state.update(_scoring_result_with_preferred("primary"))
        assert t3.event_type == TransitionType.NO_CHANGE

        t4 = state.update(_scoring_result_with_preferred("backup"))
        assert t4.event_type == TransitionType.FAILOVER
        assert t4.previous_path.name == "primary"
        assert t4.new_path.name == "backup"
        assert state.current_preferred_path.name == "backup"

        t5 = state.update(_scoring_result_with_preferred("backup"))
        assert t5.event_type == TransitionType.NO_CHANGE
        assert state.current_preferred_path.name == "backup"

        t6 = state.update(_scoring_result_no_preferred())
        assert t6.event_type == TransitionType.NO_AVAILABLE_PATH
        assert state.current_preferred_path is None

        t7 = state.update(_scoring_result_with_preferred("primary"))
        assert t7.event_type == TransitionType.INITIAL_SELECTION
        assert state.current_preferred_path.name == "primary"

    def test_repeated_no_change_does_not_accumulate_failovers(self) -> None:
        """100 updates with the same path → exactly 1 INITIAL_SELECTION, 99 NO_CHANGE."""
        state = PathState()
        transitions = [
            state.update(_scoring_result_with_preferred("primary"))
            for _ in range(100)
        ]
        initial = [t for t in transitions if t.event_type == TransitionType.INITIAL_SELECTION]
        no_change = [t for t in transitions if t.event_type == TransitionType.NO_CHANGE]
        failovers = [t for t in transitions if t.event_type == TransitionType.FAILOVER]

        assert len(initial) == 1
        assert len(no_change) == 99
        assert len(failovers) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Part 8: Transition result contents
# ══════════════════════════════════════════════════════════════════════════════


class TestTransitionContents:
    """PathTransition must contain correct paths and event information."""

    def test_new_scored_path_populated_on_initial_selection(self) -> None:
        state = PathState()
        sp = _make_scored_path("primary", available=True, total_score=80.0)
        result = _make_scoring_result(preferred=sp)
        transition = state.update(result)
        assert transition.new_scored_path is not None
        assert transition.new_scored_path.path.name == "primary"

    def test_new_scored_path_is_none_on_no_available_path(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        assert transition.new_scored_path is None

    def test_previous_scored_path_none_on_initial_selection(self) -> None:
        """On INITIAL_SELECTION the previous path is None, so no previous scored path."""
        state = PathState()
        sp = _make_scored_path("primary")
        transition = state.update(_make_scoring_result(preferred=sp))
        assert transition.previous_scored_path is None

    def test_transition_has_all_required_attributes(self) -> None:
        state = PathState()
        transition = state.update(_scoring_result_with_preferred("primary"))
        assert hasattr(transition, "event_type")
        assert hasattr(transition, "previous_path")
        assert hasattr(transition, "new_path")
        assert hasattr(transition, "previous_scored_path")
        assert hasattr(transition, "new_scored_path")

    def test_transition_is_immutable(self) -> None:
        """PathTransition is frozen — attribute assignment must raise."""
        state = PathState()
        transition = state.update(_scoring_result_with_preferred("primary"))
        with pytest.raises(Exception):
            transition.event_type = TransitionType.FAILOVER  # type: ignore[misc]

    def test_transition_event_type_is_enum(self) -> None:
        state = PathState()
        transition = state.update(_scoring_result_with_preferred("primary"))
        assert isinstance(transition.event_type, TransitionType)


# ══════════════════════════════════════════════════════════════════════════════
# Part 9: PathTransition convenience helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestTransitionHelpers:
    """is_failover() and is_change() return correct booleans for each event type."""

    def _transition_with_type(self, event_type: TransitionType) -> PathTransition:
        return PathTransition(
            event_type=event_type,
            previous_path=None,
            new_path=None,
            previous_scored_path=None,
            new_scored_path=None,
        )

    def test_is_failover_true_only_for_failover(self) -> None:
        assert self._transition_with_type(TransitionType.FAILOVER).is_failover() is True
        assert self._transition_with_type(TransitionType.INITIAL_SELECTION).is_failover() is False
        assert self._transition_with_type(TransitionType.NO_CHANGE).is_failover() is False
        assert self._transition_with_type(TransitionType.NO_AVAILABLE_PATH).is_failover() is False

    def test_is_change_false_only_for_no_change(self) -> None:
        assert self._transition_with_type(TransitionType.NO_CHANGE).is_change() is False
        assert self._transition_with_type(TransitionType.FAILOVER).is_change() is True
        assert self._transition_with_type(TransitionType.INITIAL_SELECTION).is_change() is True
        assert self._transition_with_type(TransitionType.NO_AVAILABLE_PATH).is_change() is True


# ══════════════════════════════════════════════════════════════════════════════
# Part 10: reset()
# ══════════════════════════════════════════════════════════════════════════════


class TestReset:
    """reset() returns the state to its initial condition."""

    def test_reset_clears_preferred_path(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        state.reset()
        assert state.current_preferred_path is None

    def test_reset_clears_has_preferred_path(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        state.reset()
        assert state.has_preferred_path is False

    def test_update_after_reset_produces_initial_selection(self) -> None:
        """After reset(), the next update with an available path → INITIAL_SELECTION."""
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        state.update(_scoring_result_with_preferred("primary"))
        state.reset()
        transition = state.update(_scoring_result_with_preferred("primary"))
        assert transition.event_type == TransitionType.INITIAL_SELECTION

    def test_reset_on_empty_state_is_safe(self) -> None:
        """reset() on a freshly-created state must not raise."""
        state = PathState()
        state.reset()  # should not raise
        assert state.current_preferred_path is None


# ══════════════════════════════════════════════════════════════════════════════
# Part 11: __repr__ and __str__ do not raise
# ══════════════════════════════════════════════════════════════════════════════


class TestStringRepresentations:
    def test_path_state_repr_does_not_raise(self) -> None:
        state = PathState()
        r = repr(state)
        assert isinstance(r, str)

    def test_path_state_repr_after_update_does_not_raise(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        r = repr(state)
        assert "primary" in r

    def test_path_transition_str_does_not_raise_on_initial_selection(self) -> None:
        state = PathState()
        transition = state.update(_scoring_result_with_preferred("primary"))
        s = str(transition)
        assert isinstance(s, str)
        assert "initial_selection" in s

    def test_path_transition_str_does_not_raise_on_failover(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_with_preferred("backup"))
        s = str(transition)
        assert "failover" in s

    def test_path_transition_str_does_not_raise_on_no_available(self) -> None:
        state = PathState()
        state.update(_scoring_result_with_preferred("primary"))
        transition = state.update(_scoring_result_no_preferred())
        s = str(transition)
        assert "no_available_path" in s

    def test_transition_type_enum_has_string_value(self) -> None:
        assert TransitionType.FAILOVER.value == "failover"
        assert TransitionType.NO_CHANGE.value == "no_change"
        assert TransitionType.INITIAL_SELECTION.value == "initial_selection"
        assert TransitionType.NO_AVAILABLE_PATH.value == "no_available_path"
