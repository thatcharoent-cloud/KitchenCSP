"""Solver outcome types for KitchenCSP.

Keeping the four outcome statuses explicit prevents the classic experimental
mistake of treating a greedy failure (or a timeout) as a proof of infeasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SolveStatus(str, Enum):
    """Outcome of running one algorithm on one instance.

    Values are snake_case strings so they serialize cleanly to CSV/JSON.
    """

    FEASIBLE = "feasible"
    """A complete schedule satisfying every constraint was found."""

    INFEASIBLE = "infeasible"
    """PROVEN infeasible: produced only by exhaustive CSP search reaching an
    empty tree, or by detecting an empty initial domain. Greedy runs and
    timed-out runs must never emit this."""

    UNSOLVED_GREEDY = "unsolved_greedy"
    """A greedy pass failed to place every dish. Does NOT prove infeasibility
    -- a different ordering (or CSP) may still succeed."""

    TIMEOUT = "timeout"
    """Search hit the time limit before deciding. Classify as unknown."""


@dataclass(frozen=True)
class SolverResult:
    """What one algorithm produced for one instance.

    Attributes:
        algorithm: Human-readable name ("fcfs", "edf", "backtracking",
            "mrv_forward_checking").
        status: One of the SolveStatus values above.
        schedule: Mapping dish_id -> start_minute for FEASIBLE results,
            otherwise None.
        runtime_ms: Wall-clock runtime in milliseconds (float).
        nodes_visited: Decision points entered. For CSP solvers: recursive
            search calls that select the next dish (the base case is not
            counted). For greedy: dishes attempted by the placement loop.
        attempted_assignments: Candidate start minutes tested. For CSP
            solvers: values drawn from a working domain. For greedy: start
            minutes scanned to find the earliest valid start (including the
            accepted one).

    Search-counter rules (identical across all solvers):
      - CSP: entering a decision node increments nodes_visited by 1. Greedy:
        each dish attempt in the placement loop increments it by 1.
      - Every candidate start time checked against current placements
        increments attempted_assignments by 1.
    The same rules are applied identically within each solver family so the
    numbers are comparable.
    """

    algorithm: str
    status: SolveStatus
    schedule: Optional[dict[int, int]] = None
    runtime_ms: float = 0.0
    nodes_visited: int = 0
    attempted_assignments: int = 0

    @property
    def has_schedule(self) -> bool:
        return self.schedule is not None
