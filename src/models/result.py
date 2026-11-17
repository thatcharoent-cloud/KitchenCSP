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
        nodes_visited: CSP only -- number of times the recursive search
            function was entered. 0 for greedy.
        attempted_assignments: CSP only -- number of candidate start values
            actually tested. 0 for greedy.

    Search-counter rules:
      - If a recursive CSP search call is entered, nodes_visited += 1.
      - Every time a value is drawn from a working domain and checked,
        attempted_assignments += 1.
    The same rules are applied identically to both CSP methods so the numbers
    are comparable.
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
