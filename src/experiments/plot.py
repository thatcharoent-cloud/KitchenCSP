"""Draw report figures from results/summary.csv.

Run from the project root (after src.experiments.summarize):
    python3 -m src.experiments.plot

Needs matplotlib (python3 -m pip install matplotlib). Writes PNG files to
results/figures/:
  success_<mode>.png   success rate vs number of dishes, one panel per tightness
  runtime_<mode>.png   median runtime vs number of dishes (log scale)
"""

from __future__ import annotations

import csv
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # draw to files, no window needed
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is needed for plots: python3 -m pip install matplotlib"
    ) from exc

from src.generation.suite import SIZES, TIGHTNESS

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

# Each algorithm always keeps the same color and marker in every figure.
STYLE = {
    "fcfs": {"color": "#2a78d6", "marker": "o", "label": "FCFS (greedy)"},
    "edf": {"color": "#eb6834", "marker": "s", "label": "EDF (greedy)"},
    "backtracking": {"color": "#1baf7a", "marker": "^", "label": "Backtracking"},
    "mrv_forward_checking": {
        "color": "#eda100",
        "marker": "D",
        "label": "MRV + forward checking",
    },
}
INK = "#3d3d3a"
GRID = "#e4e3dd"


def load_summary(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_xticks(SIZES)


def _series(rows, mode, tightness, algorithm, field):
    xs, ys = [], []
    for n in SIZES:
        for r in rows:
            if (
                r["mode"] == mode
                and r["tightness"] == tightness
                and r["algorithm"] == algorithm
                and int(r["n_dishes"]) == n
                and r[field] != ""
            ):
                xs.append(n)
                ys.append(float(r[field]))
    return xs, ys


def plot_metric(rows, mode, field, ylabel, title, out, log=False) -> None:
    algorithms = [a for a in STYLE if any(r["algorithm"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(TIGHTNESS), figsize=(11, 3.6), sharey=True)
    for ax, tightness in zip(axes, TIGHTNESS):
        for alg in algorithms:
            xs, ys = _series(rows, mode, tightness, alg, field)
            if field.startswith("success"):
                ys = [y * 100 for y in ys]
            s = STYLE[alg]
            ax.plot(
                xs,
                ys,
                color=s["color"],
                marker=s["marker"],
                markersize=6,
                linewidth=2,
                label=s["label"],
            )
        ax.set_title(f"{tightness} deadlines", color=INK, fontsize=10)
        ax.set_xlabel("number of dishes", color=INK, fontsize=9)
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-3, 103)
        _style_axes(ax)
    axes[0].set_ylabel(ylabel, color=INK, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(algorithms),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(title, color=INK, fontsize=11, y=1.08)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    rows = load_summary(RESULTS_DIR / "summary.csv")
    for mode in ("feasible", "random"):
        if not any(r["mode"] == mode for r in rows):
            continue
        name = "guaranteed-feasible" if mode == "feasible" else "random"
        plot_metric(
            rows,
            mode,
            "success_rate_on_solvable",
            "schedules found (% of solvable)",
            f"Success rate on {name} instances",
            FIG_DIR / f"success_{mode}.png",
        )
        plot_metric(
            rows,
            mode,
            "median_runtime_ms",
            "median runtime (ms, log scale)",
            f"Runtime on {name} instances",
            FIG_DIR / f"runtime_{mode}.png",
            log=True,
        )


if __name__ == "__main__":
    main()
