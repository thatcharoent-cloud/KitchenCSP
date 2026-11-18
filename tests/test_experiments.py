"""Unit tests for the experiment runner and summarizer."""

import csv

from src.experiments.run import ALGORITHMS, check_result, main, run_experiment, run_one
from src.experiments.summarize import classify_random_instances, summarize
from src.generation.suite import generate_suite, save_suite
from src.models.dish import Dish
from src.models.result import SolverResult, SolveStatus


def _write_small_suite(tmp_path):
    for mode in ("random", "feasible"):
        save_suite(generate_suite(mode), tmp_path / f"instances_{mode}.json")


def test_runner_produces_one_row_per_instance_and_algorithm(tmp_path):
    _write_small_suite(tmp_path)
    rows, problems = run_experiment(
        ["random", "feasible"],
        ["fcfs", "edf", "backtracking"],
        timeout_ms=50,
        repeats=1,
        limit=1,
        data_dir=tmp_path,
        progress=False,
    )
    # 12 cells x 1 instance x 2 modes x 3 algorithms
    assert len(rows) == 12 * 2 * 3
    assert problems == []
    assert {r["algorithm"] for r in rows} == {"fcfs", "edf", "backtracking"}


def test_every_feasible_row_is_validated(tmp_path):
    _write_small_suite(tmp_path)
    rows, _ = run_experiment(
        ["feasible"], ["edf"], 50, 1, limit=2, data_dir=tmp_path, progress=False
    )
    for r in rows:
        if r["status"] == "feasible":
            assert r["valid"] == "yes"
        else:
            assert r["valid"] == ""


def test_greedy_never_reports_infeasible_or_timeout(tmp_path):
    _write_small_suite(tmp_path)
    rows, _ = run_experiment(
        ["random"], ["fcfs", "edf"], 50, 1, limit=2, data_dir=tmp_path, progress=False
    )
    assert {r["status"] for r in rows} <= {"feasible", "unsolved_greedy"}


def test_repeats_give_several_runtimes_but_timeout_is_not_repeated():
    dishes = [Dish(1, "wok", 0, 2, 5)]
    result, runtimes = run_one(ALGORITHMS["edf"], dishes, 100, repeats=3)
    assert result.status == SolveStatus.FEASIBLE
    assert len(runtimes) == 3

    def always_timeout(dishes, timeout_ms):
        return SolverResult("fake", SolveStatus.TIMEOUT, runtime_ms=timeout_ms)

    _, runtimes = run_one(always_timeout, dishes, 100, repeats=3)
    assert runtimes == [100]


def test_check_result_flags_invalid_schedule_and_false_infeasible():
    dishes = [Dish(1, "wok", 0, 2, 5), Dish(2, "wok", 0, 2, 5)]
    record = {
        "mode": "feasible",
        "instance_id": "x",
        "dishes": dishes,
        "planted_schedule": {1: 0, 2: 2},
    }
    overlapping = SolverResult("fake", SolveStatus.FEASIBLE, schedule={1: 0, 2: 1})
    valid, problems = check_result(record, overlapping)
    assert valid == "no" and problems

    wrong_claim = SolverResult("backtracking", SolveStatus.INFEASIBLE)
    _, problems = check_result(record, wrong_claim)
    assert any("planted" in p for p in problems)


def test_classification_never_uses_greedy_failure_as_proof():
    rows = [
        {"mode": "random", "instance_id": "n5-tight-00", "algorithm": "fcfs",
         "status": "unsolved_greedy", "valid": ""},
        {"mode": "random", "instance_id": "n5-tight-00", "algorithm": "backtracking",
         "status": "timeout", "valid": ""},
        {"mode": "random", "instance_id": "n5-tight-01", "algorithm": "backtracking",
         "status": "infeasible", "valid": ""},
        {"mode": "random", "instance_id": "n5-tight-02", "algorithm": "edf",
         "status": "feasible", "valid": "yes"},
    ]
    classes = classify_random_instances(rows)
    assert classes == {
        "n5-tight-00": "unknown",
        "n5-tight-01": "proven_infeasible",
        "n5-tight-02": "known_feasible",
    }


def test_main_writes_csv(tmp_path, monkeypatch):
    _write_small_suite(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.csv"
    code = main(["--limit", "1", "--timeout-ms", "50", "--repeats", "1",
                 "--algorithms", "edf", "--out", str(out)])
    assert code == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 24
    summary, class_rows = summarize(
        [dict(r, n_dishes=int(r["n_dishes"]), runtime_ms=float(r["runtime_ms"]),
              nodes_visited=int(r["nodes_visited"])) for r in rows]
    )
    assert len(summary) == 24  # 2 modes x 12 cells x 1 algorithm
    assert len(class_rows) == 12
