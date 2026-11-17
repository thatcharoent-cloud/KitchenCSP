"""Independent schedule validator for KitchenCSP.

Why "independent"? The validator must NOT import any solver helper (e.g.
greedy._overlaps). It re-derives every check straight from the formal model
in README.md so a bug in a solver cannot hide behind an identical
bug in its own checker. Every algorithm result (fcfs, edf, backtracking,
mrv_fc) and every experiment run goes through this function.

Formal spec being checked, for a schedule mapping dish_id -> start_minute:
  1. A schedule was actually produced (not None).
  2. The schedule covers exactly the input dishes: no missing dish, no
     unknown ids, and every start time is a real integer (mind: in Python
     `True` is an int, so booleans must be rejected).
  3. start_i >= arrival_i  (starts no earlier than the order arrives)
  4. start_i + duration_i <= deadline_i  (finishes no later than deadline)
  5. For any i, j on the SAME station:
     (start_i + duration_i <= start_j) OR (start_j + duration_j <= start_i)
     i.e. adjacent end-to-start touching IS allowed, overlap is not.
"""

from __future__ import annotations

from src.models.dish import Dish


def find_violations(
    dishes: list[Dish], schedule: dict[int, int] | None
) -> list[str]:
    """Return every constraint violation as a human-readable string.

    The returned list is empty iff the schedule is valid. Messages look like
    "dish 2 starts at 3, before arrival 5" so failed tests/debugging are easy.
    """
    violations: list[str] = []

    if schedule is None:
        return ["no schedule was produced (schedule is None)"]

    dish_by_id = {d.id: d for d in dishes}

    if len(dish_by_id) != len(dishes):
        seen: set[int] = set()
        duplicates: set[int] = set()
        for d in dishes:
            if d.id in seen:
                duplicates.add(d.id)
            seen.add(d.id)
        violations.append(f"duplicate dish id(s): {sorted(duplicates)}")
        return violations

    for dish_id in sorted(set(dish_by_id) - set(schedule)):
        violations.append(f"dish {dish_id} has no start time")
    for dish_id in sorted(set(schedule) - set(dish_by_id)):
        violations.append(f"schedule has a start time for unknown dish {dish_id}")

    valid_starts: dict[int, int] = {}

    for dish_id, start in schedule.items():
        dish = dish_by_id.get(dish_id)

        if dish is None:
            continue
        if not isinstance(start, int) or isinstance(start, bool):
            violations.append(f"dish {dish_id} has non-integer start time {start}")
            continue

        valid_starts[dish_id] = start

        if start < dish.arrival:
            violations.append(f"dish {dish_id} start at {start}, before arrival {dish.arrival}")
        if start + dish.duration > dish.deadline:
            violations.append(f"dish {dish_id} finishes at {start + dish.duration}, "
                              f"after deadline {dish.deadline}")

    for i in range(len(dishes)):
        for j in range(i+1, len(dishes)):
            a, b = dishes[i], dishes[j]

            if a.station != b.station:
                continue

            if a.id not in valid_starts or b.id not in valid_starts:
                continue

            a_start = valid_starts[a.id]
            a_end = a_start + a.duration

            b_start = valid_starts[b.id]
            b_end = b_start + b.duration

            if a_start < b_end and b_start < a_end:
                violations.append(f"dishes {a.id} and {b.id} overlap on {a.station}")

    return violations


def validate_schedule(
    dishes: list[Dish], schedule: dict[int, int] | None
) -> bool:
    """Return True iff `schedule` satisfies every constraint.

    Directly usable in tests:
        result = fcfs(dishes)
        assert validate_schedule(dishes, result.schedule)
    """
    return len(find_violations(dishes, schedule)) == 0
