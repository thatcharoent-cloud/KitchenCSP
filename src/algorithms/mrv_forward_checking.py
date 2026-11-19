"""Backtracking CSP solver with MRV variable selection and forward checking.

Implements the enhanced search described in README.md:
  * MRV (minimum remaining values): at every decision node, pick the
    unassigned dish with the smallest current working domain, breaking ties
    deterministically by dish id.
  * Forward checking: after assigning `dish = start`, prune every conflicting
    start value from the working domains of unassigned dishes on the same
    station. If any domain becomes empty (wipeout), the assignment is
    rejected without recursing deeper.
  * Exact domain restoration: pruned domains are saved in an undo trail and
    restored when backtracking, so sibling branches always start from the
    same domain state.

Search-counter conventions (identical to backtracking.py so the numbers are
comparable):
  * nodes_visited counts DECISION nodes; the base case is not counted.
  * attempted_assignments counts every candidate start value tested.

Outcomes:
  * FEASIBLE   -- complete schedule found.
  * INFEASIBLE -- search tree exhausted (proven), or an initial domain is
                  empty before any search.
  * TIMEOUT    -- wall-clock deadline exceeded; result is unknown.
"""

from __future__ import annotations

import time

from src.models.dish import Dish, ensure_unique_ids
from src.models.result import SolverResult, SolveStatus


class _SearchTimeout(Exception):
    """Internal signal: the wall-clock deadline passed mid-search."""


def mrv_forward_checking(
    dishes: list[Dish],
    timeout_ms: float | None = 5000.0,
) -> SolverResult:
    """Search for any feasible schedule using MRV + forward checking.

    Args:
        dishes: Dishes to schedule (ids must be unique).
        timeout_ms: Wall-clock limit in milliseconds. Pass None to disable
            the timeout (useful for deterministic exhaustive unit tests).

    Returns:
        A SolverResult with algorithm="mrv_forward_checking"; see module
        docstring for the meaning of each status.
    """
    start_ns = time.perf_counter_ns()

    def elapsed_ms() -> float:
        return (time.perf_counter_ns() - start_ns) / 1_000_000

    ensure_unique_ids(dishes, "mrv_forward_checking")

    ordered = sorted(dishes, key=lambda d: d.id)
    n = len(ordered)

    # Fast-fail on any empty initial domain (same rule as backtracking).
    if any(len(d.domain) == 0 for d in ordered):
        return SolverResult(
            algorithm="mrv_forward_checking",
            status=SolveStatus.INFEASIBLE,
            runtime_ms=elapsed_ms(),
        )

    # Working domains, ascending start times. The Dish blueprint is never
    # mutated; forward checking may shrink these working lists.
    domains: dict[int, list[int]] = {d.id: list(d.domain) for d in ordered}

    # Same-station peers only: different stations never conflict, so forward
    # checking never needs to inspect them.
    peers: dict[int, list[Dish]] = {
        d.id: [o for o in ordered if o.id != d.id and o.station == d.station]
        for d in ordered
    }
    deadline_ns: int | None = None
    if timeout_ms is not None:
        deadline_ns = start_ns + int(max(0.0, timeout_ms) * 1_000_000)

    nodes = 0
    assignments = 0
    schedule: dict[int, int] = {}

    def check_timeout() -> None:
        if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
            raise _SearchTimeout()

    def search() -> bool:
        nonlocal nodes, assignments
        if len(schedule) == n:
            return True
        nodes += 1
        check_timeout()

        # MRV: smallest current domain first; deterministic id tie-break.
        dish = min(
            (d for d in ordered if d.id not in schedule),
            key=lambda d: (len(domains[d.id]), d.id),
        )

        # Snapshot: this dish's own domain is never modified by forward
        # checking, but keep the iteration stable for clarity.
        for start in list(domains[dish.id]):
            assignments += 1
            saved: list[tuple[int, list[int]]] = []
            schedule[dish.id] = start

            # Forward check: prune conflicting values from unassigned peers.
            wiped_out = False
            for other in peers[dish.id]:
                if other.id in schedule:
                    continue
                kept = [
                    t
                    for t in domains[other.id]
                    if not (
                        start < t + other.duration
                        and t < start + dish.duration
                    )
                ]
                if len(kept) != len(domains[other.id]):
                    saved.append((other.id, domains[other.id]))
                    domains[other.id] = kept
                if not kept:
                    wiped_out = True
                    break
            if not wiped_out and search():
                return True

            # Exact restoration, then try the next start value.
            del schedule[dish.id]
            for other_id, previous in saved:
                domains[other_id] = previous
            check_timeout()

        return False

    try:
        feasible = search()
    except _SearchTimeout:
        return SolverResult(
            algorithm="mrv_forward_checking",
            status=SolveStatus.TIMEOUT,
            runtime_ms=elapsed_ms(),
            nodes_visited=nodes,
            attempted_assignments=assignments,
        )

    if feasible:
        return SolverResult(
            algorithm="mrv_forward_checking",
            status=SolveStatus.FEASIBLE,
            schedule=dict(schedule),
            runtime_ms=elapsed_ms(),
            nodes_visited=nodes,
            attempted_assignments=assignments,
        )

    return SolverResult(
        algorithm="mrv_forward_checking",
        status=SolveStatus.INFEASIBLE,
        runtime_ms=elapsed_ms(),
        nodes_visited=nodes,
        attempted_assignments=assignments,
    )


# Convenience alias matching the name used in the experiment runner.
mrv_fc = mrv_forward_checking

