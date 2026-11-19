"""Greedy baseline schedulers: FCFS and EDF.

Implements the two greedy baselines: deterministic
sort keys, ascending-domain earliest-feasible-start placement, and one-pass
non-revising assignment.

Conventions:
  * Tie-break all primary sort keys by integer dish `.id`.
  * Value ordering: candidate start minutes ascending, drawn from `dish.domain`.
  * "Earliest feasible start" = smallest t in dish.domain that overlaps NO
    already-placed dish on the same station. It is NOT simply the first free
    minute after the last scheduled dish.
  * Greedy never revises earlier placements. It never emits INFEASIBLE or
    TIMEOUT; failure to place every dish yields UNSOLVED_GREEDY.
  * Search counters (comparable with the CSP solvers): nodes_visited counts
    the dishes attempted by the one-pass placement loop (including a failing
    dish); attempted_assignments counts the start minutes scanned for each
    dish, including the accepted one.
"""

from __future__ import annotations

from src.models.dish import Dish, ensure_unique_ids
from src.models.result import SolverResult, SolveStatus

import time


def _overlaps(previous: Dish, prev_start: int, dish: Dish, start: int) -> bool:
    """Return True if the interval [start, start + dish.duration) overlaps
    [prev_start, prev_start + previous.duration).

    Both dishes are guaranteed to be on the same station by the caller.
    NOTE: touching edges do NOT overlap: [0, 2) and [2, 4) are compatible.
    """

    return start < prev_start + previous.duration and prev_start < start + dish.duration


def _earliest_valid_start(
    dish: Dish, schedule: dict[int, int], dish_by_id: dict[int, Dish]
) -> tuple[int | None, int]:
    """Return (best_start, tested).

    best_start is the smallest t in dish.domain that overlaps no scheduled
    dish on the same station, or None if the dish cannot be placed anywhere.
    tested is the number of candidate start minutes scanned (including the
    accepted one, when a start is found).

    `schedule` maps dish_id -> start_minute for dishes already placed.
    """
    tested = 0
    for t in dish.domain:
        tested += 1
        conflict = False

        for placed_id, placed_start in schedule.items():

            placed_dish = dish_by_id[placed_id]

            if placed_dish.station != dish.station:
                continue

            if _overlaps(placed_dish, placed_start, dish, t):
                conflict = True
                break
        
        if not conflict:
            return t, tested

    return None, tested


def _place_all(order: list[Dish], algorithm: str, start_ns: int) -> SolverResult:
    """One-pass, non-revising placement of every dish in `order`.

    Returns:
        FEASIBLE  with the completed schedule on success,
        UNSOLVED_GREEDY with schedule=None on the first dish that has no
        valid start (this does NOT prove the instance infeasible).
    """

    ensure_unique_ids(order, algorithm)

    schedule = {}

    dish_by_id = {d.id: d for d in order}

    nodes_visited = 0
    attempted = 0

    for dish in order:
        nodes_visited += 1

        t, tested = _earliest_valid_start(dish, schedule, dish_by_id)
        attempted += tested
        if t is None:
            return SolverResult(
                algorithm=algorithm,
                status=SolveStatus.UNSOLVED_GREEDY,
                runtime_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
                nodes_visited=nodes_visited,
                attempted_assignments=attempted,
            )

        schedule[dish.id] = t

    return SolverResult(
        algorithm=algorithm,
        status=SolveStatus.FEASIBLE,
        schedule=schedule,
        runtime_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
        nodes_visited=nodes_visited,
        attempted_assignments=attempted,
    )


def fcfs(dishes: list[Dish]) -> SolverResult:
    """First-Come, First-Served.

    Processing order: sorted(dishes, key=lambda d: (d.arrival, d.id)).
    Place each dish at its earliest feasible start without revising earlier
    placements (delegate to _place_all).
    """

    start_ns = time.perf_counter_ns()

    order = sorted(dishes, key=lambda d: (d.arrival, d.id))
    return _place_all(order, "fcfs", start_ns)


def edf(dishes: list[Dish]) -> SolverResult:
    """Earliest Deadline First.

    Processing order: sorted(dishes, key=lambda d: (d.deadline, d.arrival, d.id)).
    Place each dish at its earliest feasible start without revising earlier
    placements (delegate to _place_all).
    """

    start_ns = time.perf_counter_ns()
    
    order = sorted(dishes, key=lambda d: (d.deadline, d.arrival, d.id))
    return _place_all(order, "edf", start_ns)
