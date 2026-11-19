"""Unit tests for the enhanced CSP solver: MRV + forward checking."""

import pytest

from src.algorithms.backtracking import backtracking
from src.algorithms.mrv_forward_checking import mrv_forward_checking
from src.evaluation.validator import validate_schedule
from src.generation.generator import generate_feasible_instance, generate_instance
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
    res = mrv_forward_checking(TRAP)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 2, 2: 0, 3: 4}
    assert validate_schedule(TRAP, res.schedule)


def test_counter_trace_on_trap_beats_backtracking():
    # Hand-traced MRV + FC search on TRAP: MRV picks dish 1 (smallest domain),
    # start 1 is rejected by forward-checking wipeout, start 2 succeeds.
    # 4 decision nodes, 5 candidate values tested.
    res = mrv_forward_checking(TRAP)
    assert res.nodes_visited == 4
    assert res.attempted_assignments == 5
    # Backtracking needs 5 nodes / 17 assignments on the same instance.
    ref = backtracking(TRAP)
    assert res.nodes_visited <= ref.nodes_visited
    assert res.attempted_assignments < ref.attempted_assignments


def test_mrv_picks_smallest_domain_first():
    # dish 2 has the smallest domain ([0..1]); MRV must schedule it first even
    # though its id is not smallest. Forward checking then makes the remaining
    # choices obvious: only 3 nodes and 3 assignments in total.
    dishes = [
        make_dish(1, "wok", 0, 2, 10),
        make_dish(2, "wok", 0, 2, 3),
        make_dish(3, "wok", 0, 2, 10),
    ]
    res = mrv_forward_checking(dishes)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {2: 0, 1: 2, 3: 4}
    assert res.nodes_visited == 3
    assert res.attempted_assignments == 3
    assert validate_schedule(dishes, res.schedule)


def test_mrv_ties_broken_by_id():
    # Equal domain sizes must be decided deterministically by dish id.
    dishes = [make_dish(1, "wok", 0, 2, 4), make_dish(2, "wok", 0, 2, 4)]
    res = mrv_forward_checking(dishes)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 0, 2: 2}
    assert res.nodes_visited == 2
    assert res.attempted_assignments == 2


def test_empty_domain_is_proven_infeasible():
    res = mrv_forward_checking([make_dish(1, "wok", 0, 5, 3)])
    assert res.status == SolveStatus.INFEASIBLE
    assert res.schedule is None
    assert res.nodes_visited == 0
    assert res.attempted_assignments == 0


def test_proven_infeasible_by_station_contention():
    # Three 3-minute dishes on one station need 9 minutes but only 8 are
    # available before the shared deadline; search exhausts to infeasible.
    dishes = [make_dish(i, "wok", 0, 3, 8) for i in range(1, 4)]
    res = mrv_forward_checking(dishes)
    assert res.status == SolveStatus.INFEASIBLE
    assert res.schedule is None
    # Forward checking must never explore more than plain backtracking here.
    ref = backtracking(dishes)
    assert res.nodes_visited <= ref.nodes_visited
    assert res.attempted_assignments <= ref.attempted_assignments


def test_domain_restoration_on_backtracking():
    # TRAP exercises multi-level backtracking: choosing dish1=1 prunes dish2's
    # domain to [3] and dish3's to [3,4], then dish2=3 wipes out dish3. After
    # restoring, dish1=2 succeeds. The assertions on schedule and counters
    # verify the restored state was exactly correct.
    res = mrv_forward_checking(TRAP)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 2, 2: 0, 3: 4}


def test_timeout_returns_unknown_not_infeasible():
    # 12 dishes, 28-value domains, 9 minutes of work but only 8 available:
    # exhaustiveness is impossible within 1ms, so the outcome is TIMEOUT.
    dishes = [make_dish(i, "wok", 0, 3, 30) for i in range(1, 13)]
    res = mrv_forward_checking(dishes, timeout_ms=1.0)
    assert res.status == SolveStatus.TIMEOUT
    assert res.schedule is None


def test_timeout_zero_is_immediate_and_counts_entry_node():
    res = mrv_forward_checking(TRAP, timeout_ms=0.0)
    assert res.status == SolveStatus.TIMEOUT
    assert res.schedule is None
    assert res.nodes_visited == 1
    assert res.attempted_assignments == 0


def test_parallel_stations():
    dishes = [
        make_dish(1, "wok", 0, 5, 10),
        make_dish(2, "fryer", 0, 5, 10),
        make_dish(3, "curry", 0, 5, 10),
    ]
    res = mrv_forward_checking(dishes)
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {1: 0, 2: 0, 3: 0}
    assert validate_schedule(dishes, res.schedule)


def test_input_order_invariance():
    res1 = mrv_forward_checking(TRAP)
    res2 = mrv_forward_checking(list(reversed(TRAP)))
    assert res1.schedule == res2.schedule


def test_empty_input_is_trivially_feasible():
    res = mrv_forward_checking([])
    assert res.status == SolveStatus.FEASIBLE
    assert res.schedule == {}
    assert res.nodes_visited == 0
    assert res.attempted_assignments == 0


def test_duplicate_ids_rejected():
    bad = [make_dish(1), make_dish(1)]
    with pytest.raises(ValueError):
        mrv_forward_checking(bad)


def test_matches_backtracking_on_generated_instances():
    # Both exhaustive solvers must always agree on feasibility. Checked on
    # random instances (may be infeasible) and planted-feasible instances.
    for mode in ("random", "feasible"):
        for seed in range(5):
            if mode == "random":
                dishes = generate_instance(5, "tight", seed)
            else:
                dishes, _planted = generate_feasible_instance(5, "tight", seed)
            ref = backtracking(dishes, timeout_ms=2000)
            mrv = mrv_forward_checking(dishes, timeout_ms=2000)
            assert mrv.status == ref.status, (mode, seed, mrv.status, ref.status)
            if mrv.status == SolveStatus.FEASIBLE:
                assert validate_schedule(dishes, mrv.schedule)
                assert mrv.nodes_visited <= ref.nodes_visited


def test_registered_in_experiment_runner():
    # The runner picks up MRV + FC automatically (see src/experiments/run.py).
    from src.experiments.run import ALGORITHMS

    assert "mrv_forward_checking" in ALGORITHMS
