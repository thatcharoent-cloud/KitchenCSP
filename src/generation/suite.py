"""Build, save and load the full experiment suite for KitchenCSP.

The suite is every combination of dish count x deadline tightness, with
INSTANCES_PER_CELL instances each: 4 x 3 x 20 = 240 instances.

Each instance has a fixed seed computed from its position in the grid, so
the whole suite is reproducible: running this file again always writes the
exact same instances.

Run from the project root:
    python3 -m src.generation.suite
This writes data/instances_random.json and data/instances_feasible.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.generation.generator import (
    generate_feasible_instance,
    generate_instance,
)
from src.models.dish import Dish

SIZES: tuple[int, ...] = (5, 10, 15, 20)
TIGHTNESS: tuple[str, ...] = ("tight", "moderate", "loose")
INSTANCES_PER_CELL: int = 20
MODES: tuple[str, ...] = ("random", "feasible")

DATA_DIR = Path("data")


def instance_seed(n_dishes: int, tightness: str, index: int) -> int:
    """Seed for one instance, readable from its digits.

    Example: 15 dishes, "moderate" (position 1), index 7 -> 151007.
    """
    return n_dishes * 10_000 + TIGHTNESS.index(tightness) * 1_000 + index


def generate_suite(mode: str = "feasible") -> list[dict]:
    """Generate all instances of the suite as plain records.

    Args:
        mode: "random" (may have no solution) or "feasible" (solution
            guaranteed, with the planted schedule stored).

    Each record looks like:
        {
          "instance_id": "n15-moderate-07",
          "mode": "feasible",
          "n_dishes": 15,
          "tightness": "moderate",
          "index": 7,
          "seed": 151007,
          "dishes": [Dish, ...],
          "planted_schedule": {dish_id: start, ...} or None,
        }
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    records: list[dict] = []
    for n in SIZES:
        for tightness in TIGHTNESS:
            for index in range(INSTANCES_PER_CELL):
                seed = instance_seed(n, tightness, index)
                if mode == "feasible":
                    dishes, planted = generate_feasible_instance(n, tightness, seed)
                else:
                    dishes, planted = generate_instance(n, tightness, seed), None
                records.append(
                    {
                        "instance_id": f"n{n}-{tightness}-{index:02d}",
                        "mode": mode,
                        "n_dishes": n,
                        "tightness": tightness,
                        "index": index,
                        "seed": seed,
                        "dishes": dishes,
                        "planted_schedule": planted,
                    }
                )
    return records


def _dish_to_json(dish: Dish) -> dict:
    return {
        "id": dish.id,
        "station": dish.station,
        "arrival": dish.arrival,
        "duration": dish.duration,
        "deadline": dish.deadline,
    }


def save_suite(records: list[dict], path: Path) -> None:
    """Write records to a JSON file (one instance per line, easy to diff)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        planted = r["planted_schedule"]
        row = dict(r)
        row["dishes"] = [_dish_to_json(d) for d in r["dishes"]]
        # JSON keys must be strings; convert back on load.
        row["planted_schedule"] = (
            None if planted is None else {str(k): v for k, v in planted.items()}
        )
        lines.append(json.dumps(row))
    path.write_text("[\n" + ",\n".join(lines) + "\n]\n", encoding="utf-8")


def load_suite(path: Path) -> list[dict]:
    """Read a JSON file written by save_suite back into Dish objects."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for row in rows:
        row["dishes"] = [Dish(**d) for d in row["dishes"]]
        planted = row["planted_schedule"]
        row["planted_schedule"] = (
            None if planted is None else {int(k): v for k, v in planted.items()}
        )
        records.append(row)
    return records


def main() -> None:
    for mode in MODES:
        records = generate_suite(mode)
        path = DATA_DIR / f"instances_{mode}.json"
        save_suite(records, path)
        print(f"wrote {len(records)} {mode} instances to {path}")


if __name__ == "__main__":
    main()
