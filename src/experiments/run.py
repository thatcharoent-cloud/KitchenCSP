"""Run every algorithm on every instance and save the raw results.

Run from the project root:
    python3 -m src.generation.suite        # first, create data/*.json
    python3 -m src.experiments.run         # then, run the experiment

Quick trial (first 2 instances of each cell, 0.1 s timeout):
    python3 -m src.experiments.run --limit 2 --timeout-ms 100

Output: results/raw_results.csv, one row per (instance, algorithm).

Rules this script follows (see README "Research integrity"):
  * Every algorithm sees exactly the same instances.
  * Every returned schedule is checked by the independent validator.
  * Statuses are kept separate: a greedy failure or a timeout is never
    counted as "infeasible".
  * Runtime is the MEDIAN of several repeated runs, because one run of a
    few milliseconds is noisy. A run that times out is not repeated.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

from src.algorithms.backtracking import backtracking
from src.algorithms.greedy import edf, fcfs
from src.evaluation.validator import find_violations
from src.generation.suite import DATA_DIR, MODES, load_suite
from src.models.result import SolverResult, SolveStatus

RESULTS_DIR = Path("results")
DEFAULT_TIMEOUT_MS = 5000.0
DEFAULT_REPEATS = 3

# Name -> function(dishes, timeout_ms) -> SolverResult.
# Greedy methods ignore the timeout (they never search).
Solver = Callable[[list, float], SolverResult]
ALGORITHMS: dict[str, Solver] = {
    "fcfs": lambda dishes, timeout_ms: fcfs(dishes),
    "edf": lambda dishes, timeout_ms: edf(dishes),
    "backtracking": lambda dishes, timeout_ms: backtracking(dishes, timeout_ms),
}

# MRV + forward checking is added automatically once it exists.
# Expected: src/algorithms/mrv_forward_checking.py with a function
#   mrv_forward_checking(dishes, timeout_ms) -> SolverResult
try:
    from src.algorithms.mrv_forward_checking import mrv_forward_checking

    ALGORITHMS["mrv_forward_checking"] = (
        lambda dishes, timeout_ms: mrv_forward_checking(dishes, timeout_ms)
    )
except ImportError:
    pass

CSV_COLUMNS = [
    "mode",
    "instance_id",
    "n_dishes",
    "tightness",
    "index",
    "seed",
    "algorithm",
    "status",
    "valid",
    "runtime_ms",
    "runtime_runs_ms",
    "nodes_visited",
    "attempted_assignments",
    "timeout_ms",
]


def run_one(
    solver: Solver, dishes: list, timeout_ms: float, repeats: int
) -> tuple[SolverResult, list[float]]:
    """Run a solver `repeats` times; return the first result and all runtimes.

    The solvers are deterministic, so status/schedule/counters are the same
    on every repeat; only the runtime changes. A timeout is not repeated
    (it would just cost another full timeout).
    """
    first = solver(dishes, timeout_ms)
    runtimes = [first.runtime_ms]
    if first.status != SolveStatus.TIMEOUT:
        for _ in range(repeats - 1):
            runtimes.append(solver(dishes, timeout_ms).runtime_ms)
    return first, runtimes


def check_result(record: dict, result: SolverResult) -> tuple[str, list[str]]:
    """Validate one result. Returns (valid_column_value, list_of_problems).

    valid is "yes"/"no" for results that carry a schedule, "" otherwise.
    Problems are things that mean a BUG, not a normal failure:
      * a FEASIBLE result whose schedule breaks a constraint;
      * an INFEASIBLE claim on an instance with a planted solution;
      * a greedy method claiming INFEASIBLE or TIMEOUT.
    """
    problems: list[str] = []
    where = f"{record['mode']}/{record['instance_id']}/{result.algorithm}"
    valid = ""

    if result.status == SolveStatus.FEASIBLE:
        violations = find_violations(record["dishes"], result.schedule)
        valid = "no" if violations else "yes"
        for v in violations:
            problems.append(f"{where}: invalid schedule: {v}")

    if result.status == SolveStatus.INFEASIBLE and record["planted_schedule"]:
        problems.append(f"{where}: claims infeasible, but a planted solution exists")

    if result.algorithm in ("fcfs", "edf") and result.status in (
        SolveStatus.INFEASIBLE,
        SolveStatus.TIMEOUT,
    ):
        problems.append(f"{where}: greedy must never return {result.status.value}")

    return valid, problems


def run_experiment(
    modes: list[str],
    algorithms: list[str],
    timeout_ms: float,
    repeats: int,
    limit: int | None = None,
    data_dir: Path = DATA_DIR,
    progress: bool = True,
) -> tuple[list[dict], list[str]]:
    """Run the experiment and return (rows, problems).

    limit: if set, only the first `limit` instances of each
    (n_dishes, tightness) cell are used -- for quick trials.
    """
    rows: list[dict] = []
    problems: list[str] = []

    for mode in modes:
        path = Path(data_dir) / f"instances_{mode}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python3 -m src.generation.suite` first."
            )
        records = load_suite(path)
        if limit is not None:
            records = [r for r in records if r["index"] < limit]

        started = time.perf_counter()
        for i, record in enumerate(records, 1):
            for name in algorithms:
                result, runtimes = run_one(
                    ALGORITHMS[name], record["dishes"], timeout_ms, repeats
                )
                valid, found = check_result(record, result)
                problems.extend(found)
                rows.append(
                    {
                        "mode": mode,
                        "instance_id": record["instance_id"],
                        "n_dishes": record["n_dishes"],
                        "tightness": record["tightness"],
                        "index": record["index"],
                        "seed": record["seed"],
                        "algorithm": name,
                        "status": result.status.value,
                        "valid": valid,
                        "runtime_ms": round(statistics.median(runtimes), 4),
                        "runtime_runs_ms": ";".join(f"{t:.4f}" for t in runtimes),
                        "nodes_visited": result.nodes_visited,
                        "attempted_assignments": result.attempted_assignments,
                        "timeout_ms": timeout_ms,
                    }
                )
            if progress:
                elapsed = time.perf_counter() - started
                print(
                    f"\r[{mode}] {i}/{len(records)} instances  ({elapsed:.0f}s)",
                    end="",
                    flush=True,
                )
        if progress:
            print()

    return rows, problems


def save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument(
        "--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS)
    )
    parser.add_argument("--timeout-ms", type=float, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "raw_results.csv")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    print(
        f"algorithms: {', '.join(args.algorithms)} | timeout {args.timeout_ms:g} ms"
        f" | repeats {args.repeats}"
    )
    rows, problems = run_experiment(
        args.modes, args.algorithms, args.timeout_ms, args.repeats, args.limit
    )
    save_rows(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")

    if problems:
        print(f"\nWARNING: {len(problems)} problem(s) found -- these mean a bug:")
        for p in problems[:20]:
            print("  ", p)
        return 1
    print("all returned schedules passed the independent validator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
