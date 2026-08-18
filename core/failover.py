"""
core/failover.py — Preferred-path state management and failover detection.

This module is the Step 9 "state layer". It sits above the Step 8 scoring
engine and maintains the currently-preferred logical path across successive
monitoring cycles. When the preferred path changes, it classifies and records
the type of transition that occurred.

Architecture overview
─────────────────────
Dependency direction is strictly one-way:

    failover.py (state layer)
        ↓  reads from
    scoring.py (ScoringResult, ScoredPath)
        ↓  reads from
    monitor.py (PathMetrics, MonitorResult)
        ↓  reads from
    loss.py, stats.py, throughput.py, path.py (primitives)

failover.py does NOT import or call any networking primitives.
It performs no I/O. It is pure state management over already-computed results.

What this module does
─────────────────────
1. Remembers the currently-preferred path (may be None).
2. Accepts a new ScoringResult from the scoring engine.
3. Compares the new preferred path to the previously-remembered one.
4. Classifies the type of transition.
5. Updates the current preferred path.
6. Returns an immutable PathTransition describing what happened.

What this module does NOT do
────────────────────────────
- Does NOT modify OS routing tables
- Does NOT issue system commands (ip route, netsh, route add, etc.)
- Does NOT open sockets or perform any network I/O
- Does NOT implement VPN, tunnels, or packet forwarding
- Does NOT call FastAPI or interact with the HTTP layer
- Does NOT implement hysteresis, cooldowns, or retry policies
- Does NOT persist state to disk or a database

All state is in-memory. A new PathState() instance starts fresh.

Transition types (TransitionType enum)
───────────────────────────────────────
INITIAL_SELECTION
    Previous preferred path was None (no path had ever been selected).
    A new available path was found for the first time.
    This is expected startup behaviour — NOT a failover event.
    Example: boot → A

NO_CHANGE
    The preferred path is the same as before.
    The system is stable. No action needed.
    Example: A → A

FAILOVER
    The preferred path changed from one non-None value to a different
    non-None value. This covers two sub-cases that share the same event:
      (a) The previous path became unavailable and the scoring engine
          selected a new best available path.
      (b) The scoring engine found a new best path even though the previous
          path is still available (e.g., a newly-added path with higher score).
    Both are represented as FAILOVER because from an operational perspective
    both require the operator to know that the active path changed.
    Example: A → B

NO_AVAILABLE_PATH
    The scoring result has no available paths (preferred_path is None).
    Covers two sub-cases:
      (a) The previous preferred path became None (all paths are down).
      (b) There was no preferred path before and there is still none
          (system started with all paths unavailable).
    The distinction between (a) and (b) can be inferred from the
    transition's previous_path field (None = was already down).
    Example: A → None, or None → None

Semantic clarity: Initial selection vs Failover
───────────────────────────────────────────────
The distinction between INITIAL_SELECTION and FAILOVER is intentional.
In a production system, initial selection is an expected startup event and
typically does not warrant an alert. Failover is an unexpected operational
event (a path degraded) and may warrant alerting, logging, or human review.
Conflating the two under "FAILOVER" would generate spurious alerts on startup.

Why hysteresis is excluded
──────────────────────────
This implementation uses direct deterministic state transitions: whatever the
scoring engine says is best, the state manager accepts immediately. In a
production system this can cause "path flapping":

    Cycle N:   A=75, B=74 → preferred=A
    Cycle N+1: A=74, B=75 → preferred=B  [FAILOVER]
    Cycle N+2: A=75, B=74 → preferred=A  [FAILOVER]

Rapid alternation between nearly-equal paths generates noise and can cause
brief connectivity disruptions. Production mitigations include:
  - Minimum hold time (don't switch unless the current path has been
    preferred for at least N seconds)
  - Score margin threshold (only switch if new path scores >= M points higher)
  - Exponential backoff on rapid consecutive failovers

These add significant policy complexity and belong in a later step after the
basic state management is verified to be correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.path import Path
from core.scoring import ScoredPath, ScoringResult


# ── Transition type enum ───────────────────────────────────────────────────────


class TransitionType(Enum):
    """
    The category of state change that occurred when PathState.update() was called.

    Members
    -------
    INITIAL_SELECTION
        The first path was selected; previous state was None.
        This is expected startup behaviour, NOT a failover.

    NO_CHANGE
        The preferred path did not change. The system is stable.

    FAILOVER
        The preferred path changed from one non-None path to a different
        non-None path. This is an operational event — the active path changed.

    NO_AVAILABLE_PATH
        No path is currently available. preferred_path is None.
        Applies whether the previous preferred path was None or non-None.
    """

    INITIAL_SELECTION = "initial_selection"
    NO_CHANGE = "no_change"
    FAILOVER = "failover"
    NO_AVAILABLE_PATH = "no_available_path"


# ── Transition result ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PathTransition:
    """
    Immutable record of a single state transition produced by PathState.update().

    Every call to PathState.update() returns exactly one PathTransition, even
    when nothing changed. This makes the API uniform — the caller always has
    a structured result to inspect, log, or act upon.

    Attributes
    ----------
    event_type:
        The category of change. See TransitionType for the exhaustive list.
    previous_path:
        The Path that was preferred BEFORE this update, or None if no path
        was selected before. This is the outgoing path on a FAILOVER.
    new_path:
        The Path that is preferred AFTER this update, or None if no path is
        currently available. This is the incoming path on a FAILOVER or
        INITIAL_SELECTION.
    previous_scored_path:
        The full ScoredPath record for the previous preferred path (if it
        appeared in the new scoring result). None if the previous path no
        longer appears in the result, or if there was no previous path.
        Included so callers can inspect the previous path's current scores
        (e.g., to understand why it was demoted).
    new_scored_path:
        The full ScoredPath record for the new preferred path (if one exists).
        None if no path is currently available.

    Why include scored paths and not just Path objects?
        Path objects carry only configuration (name, host, port).
        ScoredPath carries the actual scores that caused the transition.
        An operator debugging "why did we failover?" needs to see both the
        previous path's current scores and the new path's scores — not just
        that a change occurred.

    Why is this frozen?
        A transition is a historical record. Mutating it after creation would
        make it untrustworthy as a log entry.
    """

    event_type: TransitionType
    previous_path: Path | None
    new_path: Path | None
    previous_scored_path: ScoredPath | None
    new_scored_path: ScoredPath | None

    def is_failover(self) -> bool:
        """Convenience: True if this transition is a FAILOVER."""
        return self.event_type == TransitionType.FAILOVER

    def is_change(self) -> bool:
        """
        Convenience: True if the preferred path changed in any meaningful way.

        Returns True for INITIAL_SELECTION, FAILOVER, and NO_AVAILABLE_PATH.
        Returns False for NO_CHANGE.
        """
        return self.event_type != TransitionType.NO_CHANGE

    def __str__(self) -> str:
        prev = self.previous_path.name if self.previous_path else "None"
        new = self.new_path.name if self.new_path else "None"
        return (
            f"PathTransition({self.event_type.value}: "
            f"{prev!r} → {new!r})"
        )


# ── State manager ──────────────────────────────────────────────────────────────


class PathState:
    """
    In-memory state manager for the currently-preferred logical path.

    Maintains a single piece of persistent state across calls:
        current_preferred_path : Path | None

    This is intentionally minimal. The state manager does not remember
    history, does not implement hysteresis, and does not perform I/O.

    Usage
    ─────
        from core.failover import PathState

        state = PathState()

        # After each monitoring + scoring cycle:
        transition = state.update(scoring_result)

        print(state.current_preferred_path)  # current best Path or None
        print(transition.event_type)          # what just happened
        print(transition.previous_path)       # where we came from
        print(transition.new_path)            # where we are now

    Thread safety
    ─────────────
    This implementation is NOT thread-safe. If multiple threads call update()
    concurrently, the state could be corrupted. For a single-threaded event
    loop (asyncio) the caller would use asyncio.to_thread() or similar to
    ensure only one update() runs at a time. Explicit locking is a later
    concern and intentionally out of scope for this step.

    Why a class, not a module-level variable?
        A class instance encapsulates state cleanly:
          - Multiple PathState instances can coexist (e.g., in tests)
          - State is reset simply by creating a new instance
          - The state is explicit and inspectable, not a hidden global
    """

    def __init__(self) -> None:
        """
        Initialise with no preferred path.

        The initial state is None — no path has been selected yet.
        The first call to update() that produces an available preferred path
        will produce a INITIAL_SELECTION transition.
        """
        self._current_preferred: ScoredPath | None = None

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def current_preferred_path(self) -> Path | None:
        """
        The currently-preferred Path configuration, or None if no path is selected.

        This is the Path object (name, host, port). To get the full scored
        result including scores and rank, use current_scored_path.
        """
        if self._current_preferred is None:
            return None
        return self._current_preferred.path

    @property
    def current_scored_path(self) -> ScoredPath | None:
        """
        The full ScoredPath for the currently-preferred path, or None.

        Includes the path's score breakdown (rtt_score, loss_score, etc.),
        its rank, and the original PathMetrics.
        """
        return self._current_preferred

    @property
    def has_preferred_path(self) -> bool:
        """True if a path is currently selected, False if state is None."""
        return self._current_preferred is not None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, scoring_result: ScoringResult) -> PathTransition:
        """
        Process a new ScoringResult and update the preferred-path state.

        This is the core operation. It:
          1. Reads the new preferred path from scoring_result.preferred_path.
          2. Compares it to the currently-remembered preferred path.
          3. Classifies the transition type.
          4. Updates self._current_preferred.
          5. Returns an immutable PathTransition.

        Parameters
        ----------
        scoring_result:
            The result of a score_paths() call. The preferred_path field
            is the single source of truth for what the scoring engine
            considers the current best available path.

        Returns
        -------
        PathTransition
            Always returns — never raises.
            The transition is immutable and self-describing.

        Transition classification logic
        ────────────────────────────────
        Let P = self._current_preferred (previous)
            N = scoring_result.preferred_path (new)

        Case 1: P is None, N is not None
            → INITIAL_SELECTION
            First time a path is selected.

        Case 2: P is not None, N is not None, same path name
            → NO_CHANGE
            Same path is still preferred. System is stable.
            Note: we compare by path NAME (not by Python object identity)
            because a new ScoringResult always creates new ScoredPath objects.
            Two ScoredPath objects representing the same path will have the
            same name but different object identities.

        Case 3: P is not None, N is not None, different path name
            → FAILOVER
            The preferred path changed. This covers both:
              (a) previous path became unavailable → best surviving path selected
              (b) a different path now scores higher (scores changed, all available)

        Case 4: N is None (no path available)
            → NO_AVAILABLE_PATH
            Applies regardless of whether P is None or not.
            (P was non-None: the current preferred path just went offline)
            (P was None: still no path, system was already down)

        Why compare by name?
            ScoredPath is a frozen dataclass created fresh each scoring cycle.
            Two ScoredPath objects for the same path may have different scores
            (the measured values changed) but refer to the same logical path.
            The path NAME is the stable identity defined by the operator.
            Using Python object identity (is) would incorrectly classify
            every update as a FAILOVER even when nothing changed.
        """
        previous_scored = self._current_preferred
        previous_path = previous_scored.path if previous_scored is not None else None

        new_scored = scoring_result.preferred_path
        new_path = new_scored.path if new_scored is not None else None

        # Find the previous path's current ScoredPath (if it appears in the
        # new result). This lets the caller inspect what score it now has,
        # useful for understanding "why was the previous path demoted?".
        previous_in_new: ScoredPath | None = None
        if previous_path is not None:
            for sp in scoring_result.scored_paths:
                if sp.path.name == previous_path.name:
                    previous_in_new = sp
                    break

        # ── Classify the transition ────────────────────────────────────────────
        if new_path is None:
            event_type = TransitionType.NO_AVAILABLE_PATH

        elif previous_path is None:
            # Previous was None → any available path is initial selection
            event_type = TransitionType.INITIAL_SELECTION

        elif new_path.name == previous_path.name:
            # Same path name → no change
            event_type = TransitionType.NO_CHANGE

        else:
            # Different non-None paths → failover
            event_type = TransitionType.FAILOVER

        # ── Update state ───────────────────────────────────────────────────────
        self._current_preferred = new_scored

        # ── Build and return transition ────────────────────────────────────────
        return PathTransition(
            event_type=event_type,
            previous_path=previous_path,
            new_path=new_path,
            previous_scored_path=previous_in_new,
            new_scored_path=new_scored,
        )

    def reset(self) -> None:
        """
        Reset state to the initial condition (no preferred path).

        Useful in tests and when the operator wants to force re-selection
        from scratch without creating a new PathState instance.
        After reset(), the next update() that finds an available path will
        produce an INITIAL_SELECTION transition.
        """
        self._current_preferred = None

    def __repr__(self) -> str:
        name = (
            self._current_preferred.path.name
            if self._current_preferred is not None
            else None
        )
        return f"PathState(current_preferred={name!r})"
