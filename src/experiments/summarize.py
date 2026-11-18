"""Summarize results/raw_results.csv into tables.

Run from the project root (after src.experiments.run):
    python3 -m src.experiments.summarize

Writes:
  results/summary.csv          one row per (mode, n_dishes, tightness, algorithm)
  results/random_classes.csv   for the random suite: how many instances are
                               known feasible / proven infeasible / unknown
and prints the same tables in Markdown, ready to paste into the report.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from src.generation.suite import SIZES, TIGHTNESS

RESULTS_DIR = Path("results")
CSP_ALGORITHMS = ("backtracking", "mrv_forward_checking")
ALGORITHM_ORDER = ("fcfs", "edf", "backtracking", "mrv_forward_checking")

SUMMARY_COLUMNS = [
    "mode",
    "n_dishes",
    "tightness",
    "algorithm",
    "instances",
    "feasible",
    "unsolved_greedy",
    "infeasible",
    "timeout",
    "invalid",
    "success_rate",
    "success_rate_on_solvable",
    "median_runtime_ms",
    "mean_runtime_ms",
    "median_nodes_visited",
]


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_dishes"] = int(r["n_dishes"])
        r["runtime_ms"] = float(r["runtime_ms"])
        r["nodes_visited"] = int(r["nodes_visited"])
    return rows


def classify_random_instances(rows: list[dict]) -> dict[str, str]:
    """instance_id -> "known_feasible" | "proven_infeasible" | "unknown".

    known_feasible: at least one algorithm returned a valid schedule.
    proven_infeasible: an exhaustive CSP search finished with INFEASIBLE.
    unknown: nobody found a schedule and every CSP run timed out.
    (Greedy failure never counts as proof.)
    """
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["mode"] == "random":
            by_instance[r["instance_id"]].append(r)

    classes: dict[str, str] = {}
    for instance_id, results in by_instance.items():
        if any(r["status"] == "feasible" and r["valid"] == "yes" for r in results):
            classes[instance_id] = "known_feasible"
        elif any(
            r["status"] == "infeasible" and r["algorithm"] in CSP_ALGORITHMS
            for r in results
        ):
            classes[instance_id] = "proven_infeasible"
        else:
            classes[instance_id] = "unknown"
    return classes


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    classes = classify_random_instances(rows)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["n_dishes"], r["tightness"], r["algorithm"])].append(r)

    def sort_key(key):
        mode, n, tightness, algorithm = key
        alg_rank = (
            ALGORITHM_ORDER.index(algorithm) if algorithm in ALGORITHM_ORDER else 99
        )
        return (mode, n, TIGHTNESS.index(tightness), alg_rank)

    summary: list[dict] = []
    for key in sorted(groups, key=sort_key):
        mode, n, tightness, algorithm = key
        g = groups[key]
        count = defaultdict(int)
        for r in g:
            count[r["status"]] += 1
        invalid = sum(1 for r in g if r["valid"] == "no")
        good = sum(1 for r in g if r["status"] == "feasible" and r["valid"] == "yes")

        # Solvable = instances we KNOW have a solution.
        if mode == "feasible":
            solvable = len(g)
        else:
            solvable = sum(
                1 for r in g if classes.get(r["instance_id"]) == "known_feasible"
            )

        runtimes = [r["runtime_ms"] for r in g]
        summary.append(
            {
                "mode": mode,
                "n_dishes": n,
                "tightness": tightness,
                "algorithm": algorithm,
                "instances": len(g),
                "feasible": good,
                "unsolved_greedy": count["unsolved_greedy"],
                "infeasible": count["infeasible"],
                "timeout": count["timeout"],
                "invalid": invalid,
                "success_rate": round(good / len(g), 4),
                "success_rate_on_solvable": (
                    round(good / solvable, 4) if solvable else ""
                ),
                "median_runtime_ms": round(statistics.median(runtimes), 4),
                "mean_runtime_ms": round(statistics.mean(runtimes), 4),
                "median_nodes_visited": (
                    statistics.median(r["nodes_visited"] for r in g)
                    if algorithm in CSP_ALGORITHMS
                    else ""
                ),
            }
        )

    class_rows: list[dict] = []
    if classes:
        cell_counts: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
        for instance_id, cls in classes.items():
            # instance_id looks like "n15-moderate-07"
            n_part, tightness, _ = instance_id.split("-")
            cell_counts[(int(n_part[1:]), tightness)][cls] += 1
        for n in SIZES:
            for tightness in TIGHTNESS:
                if (n, tightness) not in cell_counts:
                    continue
                c = cell_counts[(n, tightness)]
                class_rows.append(
                    {
                        "n_dishes": n,
                        "tightness": tightness,
                        "known_feasible": c["known_feasible"],
                        "proven_infeasible": c["proven_infeasible"],
                        "unknown": c["unknown"],
                    }
                )
    return summary, class_rows


def write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in columns) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize raw experiment results.")
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw_results.csv")
    args = parser.parse_args(argv)

    rows = load_rows(args.raw)
    summary, class_rows = summarize(rows)

    write_csv(summary, SUMMARY_COLUMNS, RESULTS_DIR / "summary.csv")
    print(f"wrote {RESULTS_DIR / 'summary.csv'}")
    class_columns = [
        "n_dishes",
        "tightness",
        "known_feasible",
        "proven_infeasible",
        "unknown",
    ]
    if class_rows:
        write_csv(class_rows, class_columns, RESULTS_DIR / "random_classes.csv")
        print(f"wrote {RESULTS_DIR / 'random_classes.csv'}")

    short = [
        "mode",
        "n_dishes",
        "tightness",
        "algorithm",
        "feasible",
        "unsolved_greedy",
        "infeasible",
        "timeout",
        "success_rate_on_solvable",
        "median_runtime_ms",
    ]
    print("\n## Results by algorithm\n")
    print(markdown_table(summary, short))
    if class_rows:
        print("\n## Random suite: what is known about each instance\n")
        print(markdown_table(class_rows, class_columns))


if __name__ == "__main__":
    main()
