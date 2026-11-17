"""Unit tests for the Dish model, SolveStatus/SolverResult, and the
independent schedule validator rules."""

import pytest

from src.evaluation.validator import find_violations, validate_schedule
from src.models.dish import Dish, ensure_unique_ids
from src.models.result import SolveStatus, SolverResult


def make_dish(**overrides):
    defaults = dict(id=1, station="wok", arrival=0, duration=2, deadline=10)
    defaults.update(overrides)
    return Dish(**defaults)


# --- Dish invariants -----------------------------------------------------


def test_dish_computes_slack_and_domain():
    d = make_dish()
    assert d.slack == 8
    assert d.latest_start == 8
    assert list(d.domain) == list(range(0, 9))


def test_dish_negative_slack_means_empty_domain():
    d = make_dish(arrival=5, duration=3, deadline=6)  # slack = -2
    assert d.slack == -2
    assert list(d.domain) == []


def test_dish_rejects_unknown_station():
    with pytest.raises(ValueError):
        make_dish(station="oven")


def test_dish_rejects_negative_arrival():
    with pytest.raises(ValueError):
        make_dish(arrival=-1)


def test_dish_rejects_zero_duration():
    with pytest.raises(ValueError):
        make_dish(duration=0)


def test_dish_rejects_deadline_before_arrival():
    with pytest.raises(ValueError):
        make_dish(arrival=5, deadline=3)


@pytest.mark.parametrize("bad", [True, "2", 2.5])
def test_dish_rejects_non_int_id(bad):
    with pytest.raises(ValueError):
        make_dish(id=bad)


@pytest.mark.parametrize("field", ["arrival", "duration", "deadline"])
def test_dish_rejects_bool_field(field):
    with pytest.raises(ValueError):
        make_dish(**{field: True})


def test_ensure_unique_ids_passes_for_unique():
    ensure_unique_ids([make_dish(id=1), make_dish(id=2)], "test")


def test_ensure_unique_ids_raises_on_duplicates():
    with pytest.raises(ValueError):
        ensure_unique_ids([make_dish(id=1), make_dish(id=1)], "test")


# --- SolveStatus / SolverResult ------------------------------------------


def test_status_values_match_spec():
    assert SolveStatus.FEASIBLE.value == "feasible"
    assert SolveStatus.INFEASIBLE.value == "infeasible"
    assert SolveStatus.UNSOLVED_GREEDY.value == "unsolved_greedy"
    assert SolveStatus.TIMEOUT.value == "timeout"


def test_solver_result_defaults():
    r = SolverResult(algorithm="x", status=SolveStatus.TIMEOUT)
    assert r.schedule is None
    assert r.runtime_ms == 0.0
    assert r.nodes_visited == 0
    assert r.attempted_assignments == 0
    assert not r.has_schedule


def test_solver_result_with_schedule():
    r = SolverResult(algorithm="x", status=SolveStatus.FEASIBLE, schedule={1: 0})
    assert r.has_schedule
    assert r.schedule == {1: 0}


# --- Independent validator rules -----------------------------------------

A = Dish(id=1, station="wok", arrival=0, duration=4, deadline=10)
B = Dish(id=2, station="wok", arrival=0, duration=3, deadline=10)
TRAP = [
    Dish(id=1, station="wok", arrival=1, duration=2, deadline=4),
    Dish(id=2, station="wok", arrival=0, duration=2, deadline=5),
    Dish(id=3, station="wok", arrival=0, duration=2, deadline=6),
]


def test_valid_schedule_passes():
    assert find_violations(TRAP, {1: 2, 2: 0, 3: 4}) == []
    assert validate_schedule(TRAP, {1: 2, 2: 0, 3: 4}) is True


def test_none_schedule_is_invalid():
    v = find_violations(TRAP, None)
    assert any("no schedule" in m for m in v)
    assert validate_schedule(TRAP, None) is False


def test_missing_dish_is_invalid():
    v = find_violations(TRAP, {1: 2, 2: 0})
    assert any("no start time" in m for m in v)


def test_unknown_dish_is_invalid():
    v = find_violations(TRAP, {1: 2, 2: 0, 3: 4, 99: 0})
    assert any("unknown dish 99" in m for m in v)


@pytest.mark.parametrize("bad_start", ["six", True])
def test_non_integer_start_is_invalid(bad_start):
    v = find_violations(TRAP, {1: bad_start, 2: 0, 3: 4})
    assert any("non-integer" in m for m in v)


def test_start_before_arrival_is_invalid():
    v = find_violations(TRAP, {1: 0, 2: 0, 3: 4})  # dish 1 arrives at 1
    assert any("before arrival" in m for m in v)


def test_finishes_after_deadline_is_invalid():
    v = find_violations(TRAP, {1: 3, 2: 0, 3: 4})  # dish 1 -> [3,5), deadline 4
    assert any("after deadline" in m for m in v)


def test_touching_boundaries_are_valid():
    assert find_violations([A, B], {1: 0, 2: 4}) == []


def test_overlap_is_reported():
    v = find_violations([A, B], {1: 0, 2: 3})
    assert any("overlap" in m for m in v)


def test_missing_dish_does_not_crash_pairwise_check():
    v = find_violations([A, B], {1: 0})
    assert any("no start time" in m for m in v)


def test_non_integer_start_does_not_crash_pairwise_check():
    v = find_violations([A, B], {1: "x", 2: 4})
    assert any("non-integer" in m for m in v)


def test_duplicate_ids_are_reported():
    v = find_violations([make_dish(id=1), make_dish(id=1)], {1: 0})
    assert any("duplicate" in m for m in v)


def test_empty_instance_is_valid():
    assert find_violations([], {}) == []
