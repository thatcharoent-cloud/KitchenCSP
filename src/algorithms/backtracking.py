"""Basic CSP solver: fixed-order recursive backtracking.

Implements the exhaustive backtracking search:
fixed variable order, ascending start times, conflict checks, and exhaustive
backtracking, plus a wall-clock timeout that maps to TIMEOUT (unknown) and is
never reported as a proof of infeasibility.

Search-counter conventions (identical across CSP solvers so the metrics are
comparable; see SolverResult in src/models/result.py):
  * nodes_visited counts DECISION nodes: recursive calls that select the next
    dish to assign. The base case (all dishes scheduled) is not counted.
  * attempted_assignments counts every candidate start value drawn from a
    dish's initial domain and tested for consistency.

Outcomes:
  * FEASIBLE    -- complete schedule found (passes the independent validator).
  * INFEASIBLE  -- search tree exhausted (proven), or an initial domain is
                   empty before any search.
  * TIMEOUT     -- wall-clock deadline exceeded; result is unknown.
"""

from __future__ import annotations

import time

from src.models.dish import Dish, ensure_unique_ids
from src.models.result import SolverResult, SolveStatus


class _SearchTimeout(Exception):
    """Internal signal: the wall-clock deadline passed mid-search."""


def _conflicts(
    dish: Dish,
    start: int,
    assignment: dict[int, int],
    dish_by_id: dict[int, Dish],
) -> bool:
    """True if [start, start + dish.duration) overlaps an already-assigned
    dish on the same station (different stations never conflict)."""
    for placed_id, placed_start in assignment.items():
        placed = dish_by_id[placed_id]
        if placed.station != dish.station:
            continue
        if start < placed_start + placed.duration and placed_start < start + dish.duration:
            return True
    return False


def backtracking(
    dishes: list[Dish],
    timeout_ms: float | None = 5000.0,
) -> SolverResult:
    """Search for any feasible schedule with fixed-order backtracking.

    Args:
        dishes: Dishes to schedule; processed in ascending dish.id order.
        timeout_ms: Wall-clock limit in milliseconds. Pass None to disable
            the timeout (useful for deterministic exhaustive unit tests).

    Returns:
        A SolverResult with algorithm="backtracking"; see module docstring
        for the meaning of each status.
    """
    start_ns = time.perf_counter_ns()

    def elapsed_ms() -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000

    ensure_unique_ids(dishes, "backtracking")

    ordered = sorted(dishes, key=lambda d: d.id)
    dish_by_id = {d.id: d for d in ordered}
    n = len(ordered)

    # Fast-fail on any empty initial domain: the instance is provably
    # infeasible before any assignment (matches the SolverResult docstring).
    if any(len(d.domain) == 0 for d in ordered):
        return SolverResult(
            algorithm="backtracking",
            status=SolveStatus.INFEASIBLE,
            runtime_ms=elapsed_ms(),
        )

    deadline_ns: int | None = None
    if timeout_ms is not None:
        deadline_ns = start_ns + int(max(0.0, timeout_ms) * 1_000_000)

    nodes = 0
    assignments = 0
    schedule: dict[int, int] = {}

    def check_timeout() -> None:
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            raise _SearchTimeout()

    def search(index: int) -> bool:
        nonlocal nodes, assignments
        if len(schedule) == n:
            return True
        nodes += 1
        check_timeout()
        dish = ordered[index]
        print(f"domain:{dish.domain}")
        for start in dish.domain:
            assignments += 1
            if not _conflicts(dish, start, schedule, dish_by_id):
                schedule[dish.id] = start
                if search(index + 1):
                    return True
                del schedule[dish.id]
            check_timeout()
        return False

    try:
        feasible = search(0)
    except _SearchTimeout:
        return SolverResult(
            algorithm="backtracking",
            status=SolveStatus.TIMEOUT,
            runtime_ms=elapsed_ms(),
            nodes_visited=nodes,
            attempted_assignments=assignments,
        )

    if feasible:
        print(f"schedule:{schedule}")
        return SolverResult(
            algorithm="backtracking",
            status=SolveStatus.FEASIBLE,
            schedule=dict(schedule),
            runtime_ms=elapsed_ms(),
            nodes_visited=nodes,
            attempted_assignments=assignments,
        )

    return SolverResult(
        algorithm="backtracking",
        status=SolveStatus.INFEASIBLE,
        runtime_ms=elapsed_ms(),
        nodes_visited=nodes,
        attempted_assignments=assignments,
    )
