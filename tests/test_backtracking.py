"""Unit tests for fixed-order recursive backtracking (basic CSP)."""

import pytest

from src.algorithms.backtracking import backtracking
from src.evaluation.validator import validate_schedule
from src.models.dish import Dish
from src.models.result import SolveStatus


def make_dish(id, station="wok", arrival=0, duration=2, deadline=10):
    return Dish(
        id=id, station=station, arrival=arrival, duration=duration, deadline=deadline
    )


TRAP = [
    make_dish(1, "wok", 1, 2, 4),
    make_dish(2, "wok", 0, 2, 5),
    make_dish(3, "wok", 0, 2, 6),
]


def test_solves_trap_that_fools_both_greedy():
    res = backtracking(TRAP)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 2, 2: 0, 3: 4}
    assert validate_schedule(TRAP, res.schedule)

TRAP2 = [
    make_dish(1, "wok", 0, 3, 6),
    make_dish(2, "wok", 0, 2, 2)
]

def test_solves_two_dish_case_with_urgent_second_dish():
    res = backtracking(TRAP2)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 2, 2: 0}


def test_counter_trace_on_trap():
    # Hand-traced search tree (decisions in ascending value order):
    # 5 decision nodes entered, 17 candidate values tested.
    res = backtracking(TRAP)
    assert res.nodes_visited == 5
    assert res.attempted_assignments == 17


def test_empty_domain_is_proven_infeasible():
    res = backtracking([make_dish(1, "wok", 0, 5, 3)])
    assert res.status == SolveStatus.INFEASIBLE
    assert res.schedule is None


def test_proven_infeasible_by_station_contention():
    # Three 3-minute dishes on one station need 9 minutes but only 8 are
    # available before the shared deadline; search exhausts to infeasible.
    dishes = [make_dish(i, "wok", 0, 3, 8) for i in range(1, 4)]
    res = backtracking(dishes)
    assert res.status == SolveStatus.INFEASIBLE
    assert res.schedule is None


def test_two_dish_feasible_counter_trace():
    # dish1 [0,2); dish2 earliest non-conflicting start is 2 -> [2,5).
    dishes = [make_dish(1, "wok", 0, 2, 4), make_dish(2, "wok", 0, 3, 5)]
    res = backtracking(dishes)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 0, 2: 2}
    assert res.nodes_visited == 2
    assert res.attempted_assignments == 4
    assert validate_schedule(dishes, res.schedule)


def test_two_dish_infeasible_counter_trace():
    # dish1 must start 0 -> [0,5); dish2 (arr 1, dur 5) cannot fit anywhere.
    dishes = [make_dish(1, "wok", 0, 5, 5), make_dish(2, "wok", 1, 5, 8)]
    res = backtracking(dishes)
    assert res.status == SolveStatus.INFEASIBLE
    assert res.nodes_visited == 2
    assert res.attempted_assignments == 4


def test_timeout_returns_unknown_not_infeasible():
    # 12 dishes, 28-value domains, 9 minutes of work but only 8 available:
    # exhaustiveness is impossible within 1ms, so the outcome is TIMEOUT.
    dishes = [make_dish(i, "wok", 0, 3, 30) for i in range(1, 13)]
    res = backtracking(dishes, timeout_ms=1.0)
    assert res.status == SolveStatus.TIMEOUT
    assert res.schedule is None


def test_timeout_zero_is_immediate_and_counts_entry_node():
    res = backtracking(TRAP, timeout_ms=0.0)
    assert res.status == SolveStatus.TIMEOUT
    assert res.schedule is None
    assert res.nodes_visited == 1
    assert res.attempted_assignments == 0


def test_no_timeout_disables_deadline():
    res = backtracking([make_dish(1, "wok", 0, 2, 5)], timeout_ms=None)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 0}


def test_parallel_stations():
    dishes = [
        make_dish(1, "wok", 0, 5, 10),
        make_dish(2, "fryer", 0, 5, 10),
        make_dish(3, "curry", 0, 5, 10),
    ]
    res = backtracking(dishes)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 0, 2: 0, 3: 0}
    assert validate_schedule(dishes, res.schedule)


def test_input_order_invariance():
    res1 = backtracking(TRAP)
    res2 = backtracking(list(reversed(TRAP)))
    assert res1.schedule == res2.schedule


def test_algorithm_tag():
    res = backtracking([make_dish(1, "wok", 0, 2, 5)])
    assert res.algorithm == "backtracking"


def test_empty_input_is_trivially_feasible():
    res = backtracking([])
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {}
    assert res.nodes_visited == 0
    assert res.attempted_assignments == 0


def test_duplicate_ids_rejected():
    bad = [make_dish(1), make_dish(1)]
    with pytest.raises(ValueError):
        backtracking(bad)
