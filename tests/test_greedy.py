import pytest

from src.models.dish import Dish
from src.algorithms.greedy import _earliest_valid_start

from src.models.result import SolveStatus
from src.algorithms.greedy import fcfs, edf

from src.evaluation.validator import validate_schedule, find_violations

def make_dish(id, station, arrival, duration, deadline):
    return Dish(
        id = id, 
        station = station, 
        arrival=arrival, 
        duration=duration, 
        deadline=deadline
    )

def test_empty_schedule_returns_first_domain_value():
    d = make_dish(1, "wok", arrival=2, duration=3, deadline=10)
    assert _earliest_valid_start(d, {}, {1:d}) == (2, 1)

def test_touching_invervals_are_allow():
    placed = make_dish(1, "wok", arrival=0, duration=4, deadline=10)
    new = make_dish(2, "wok", arrival=0, duration=3, deadline=20)
    dish_by_id = {1: placed, 2: new}
    assert _earliest_valid_start(new, {1:0}, dish_by_id) == (4, 5)

def test_different_station_does_not_conflict():
    placed = make_dish(1, "wok", arrival=0, duration=4, deadline=10)
    new = make_dish(2, "fryer", arrival=0, duration=3, deadline=20)
    dish_by_id = {1: placed, 2: new}
    assert _earliest_valid_start(new, {1:0}, dish_by_id) == (0, 1)

def test_fills_earlier_gap():
    placed = make_dish(1, "wok", arrival=10, duration=4, deadline=14)
    new = make_dish(2, "wok", arrival=0, duration=5, deadline=13)
    dish_by_id = {1: placed, 2: new}
    assert _earliest_valid_start(new, {1:10}, dish_by_id) == (0, 1)

def test_return_none_when_no_start_fits():
    placed = make_dish(1, "wok", arrival=0, duration=10, deadline=10)
    new = make_dish(2, "wok", arrival=1, duration=10, deadline=11)
    dish_by_id = {1: placed, 2: new}
    assert _earliest_valid_start(new, {1:0}, dish_by_id) == (None, 1)

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_single_dish_starts_at_arrival(algo):
    d = make_dish(1, "wok", arrival=3, duration=4, deadline=20)
    result = algo([d])
    assert result.status == SolveStatus.FEASIBLE
    assert result.schedule == {1: 3}

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_different_stations_run_in_parallel(algo):
    dishes = [
        make_dish(1, "wok", 0, 5, 10), 
        make_dish(2, "fryer", 0, 5, 10), 
        make_dish(3, "curry", 0, 5, 10)
    ]
    result = algo(dishes)
    assert result.status == SolveStatus.FEASIBLE
    assert result.schedule == {1: 0, 2: 0, 3: 0}

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_returned_schedule_passes_validator(algo):
    dishes = [
        make_dish(1, "wok", 0, 3, 20), 
        make_dish(2, "wok", 1, 4, 20), 
        make_dish(3, "fryer", 2, 5, 20)
    ]
    result = algo(dishes)
    assert result.status == SolveStatus.FEASIBLE
    assert validate_schedule(dishes, result.schedule) is True

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_empty_domain_is_unsolved(algo):
    d = make_dish(1, "wok", arrival=0, duration=5, deadline=3)
    result = algo([d])
    assert result.status == SolveStatus.UNSOLVED_GREEDY

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_input_order_does_not_change_result(algo):
    dishes = [
        make_dish(1, "wok", 0, 3, 20), 
        make_dish(2, "wok", 0, 3, 20), 
        make_dish(3, "wok", 1, 2, 20)
    ]
    forward = algo(dishes)
    backward = algo(list(reversed(dishes)))
    assert forward.schedule == backward.schedule

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_input_dishes_are_not_mutated(algo):
    dishes = [make_dish(1, "wok", 0, 3, 20)]
    before = list(dishes)
    algo(dishes)
    assert dishes == before

def test_edf_succeeds_where_fcfs_fails():
    dishes = [
        make_dish(1, "wok", 0, 4, 10), 
        make_dish(2, "wok", 0, 2, 3)
    ]
    assert fcfs(dishes).status ==SolveStatus.UNSOLVED_GREEDY
    edf_result = edf(dishes)
    assert edf_result.status == SolveStatus.FEASIBLE
    assert validate_schedule(dishes, edf_result.schedule)

TRAP = [
    make_dish(1, "wok", 1, 2, 4), 
    make_dish(2, "wok", 0, 2, 5), 
    make_dish(3, "wok", 0, 2, 6), 
]

def test_fcfs_fails_on_trap():
    assert fcfs(TRAP).status == SolveStatus.UNSOLVED_GREEDY

def test_edf_fails_on_trap():
    assert edf(TRAP).status == SolveStatus.UNSOLVED_GREEDY

def test_touching_interval_are_valid():
    dishes = [
        make_dish(1, "wok", 0, 4, 10), 
        make_dish(2, "wok", 0, 3, 10), 
    ]
    assert find_violations(dishes, {1:0, 2:4}) == []

def test_overlap_is_reported():
    dishes = [
        make_dish(1, "wok", 0, 4, 10), 
        make_dish(2, "wok", 0, 3, 10), 
    ]
    v = find_violations(dishes, {1:0, 2:3})
    assert any("overlap" in msg for msg in v)

@pytest.mark.parametrize("algo", [fcfs, edf])
def test_runtime_recorded_on_failure(algo):
    d = make_dish(1, "wok", 0, 5, 3)
    result = algo([d])
    print(f"result.runtime_ms: {result.runtime_ms }")
    assert result.status == SolveStatus.UNSOLVED_GREEDY
    assert isinstance(result.runtime_ms, float)
    assert result.runtime_ms > 0

def test_fcfs_ties_broken_by_id():
    dishes = [
        make_dish(2, "wok", 0, 3, 20),
        make_dish(1, "wok", 0, 3, 20),
    ]
    assert fcfs(dishes).schedule == {1: 0, 2: 3}


def test_edf_orders_by_deadline_before_id():
    dishes = [
        make_dish(1, "wok", 0, 2, 20),
        make_dish(2, "wok", 0, 2, 5),
    ]
    assert edf(dishes).schedule == {2: 0, 1: 2}


def test_edf_breaks_deadline_ties_by_arrival():
    dishes = [
        make_dish(1, "wok", 5, 2, 10),
        make_dish(2, "wok", 0, 2, 10),
    ]
    assert edf(dishes).schedule == {2: 0, 1: 5}


def test_algorithm_names_are_tagged():
    single = [make_dish(1, "wok", 0, 2, 5)]
    assert fcfs(single).algorithm == "fcfs"
    assert edf(single).algorithm == "edf"


def test_empty_input_is_trivially_feasible():
    for algo in (fcfs, edf):
        res = algo([])
        assert res.status == SolveStatus.FEASIBLE
        assert res.schedule == {}


def test_duplicate_ids_are_rejected():
    bad = [make_dish(1, "wok", 0, 2, 5), make_dish(1, "wok", 0, 2, 5)]
    with pytest.raises(ValueError):
        fcfs(bad)
    with pytest.raises(ValueError):
        edf(bad)


def test_greedy_counters_single_dish():
    single = [make_dish(1, "wok", 3, 4, 20)]
    for algo in (fcfs, edf):
        res = algo(single)
        assert res.status == SolveStatus.FEASIBLE
        assert res.nodes_visited == 1
        assert res.attempted_assignments == 1


def test_greedy_counters_on_trap_fcfs():
    # FCFS order: dish 2 (1 tested), dish 3 (3 tested), dish 1 (2 tested).
    res = fcfs(TRAP)
    assert res.status == SolveStatus.UNSOLVED_GREEDY
    assert res.nodes_visited == 3
    assert res.attempted_assignments == 6


def test_greedy_counters_on_trap_edf():
    # EDF order: dish 1 (1 tested), dish 2 (4 tested), dish 3 (5 tested).
    res = edf(TRAP)
    assert res.status == SolveStatus.UNSOLVED_GREEDY
    assert res.nodes_visited == 3
    assert res.attempted_assignments == 10


def test_greedy_counters_empty_domain_failure():
    d = make_dish(1, "wok", 0, 5, 3)
    for algo in (fcfs, edf):
        res = algo([d])
        assert res.status == SolveStatus.UNSOLVED_GREEDY
        assert res.nodes_visited == 1
        assert res.attempted_assignments == 0
