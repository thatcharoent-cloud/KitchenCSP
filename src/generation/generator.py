"""Random instance generator for KitchenCSP.

Each instance is a list of Dish objects. The same seed always gives the
same dishes, so every experiment can be reproduced exactly.
"""

from __future__ import annotations

import random

from src.models.dish import Dish, STATIONS

# Slack = deadline - arrival - duration (minutes), as defined in README.md.
SLACK_RANGES: dict[str, tuple[int, int]] = {
    "tight": (0, 3),
    "moderate": (4, 8),
    "loose": (9, 15),
}

# Cooking time of one dish, in minutes.
DURATION_RANGE: tuple[int, int] = (2, 8)

# Orders arrive within the first (ARRIVAL_MINUTES_PER_DISH * n) minutes,
# so more dishes means a longer service window, not just a bigger crowd.
ARRIVAL_MINUTES_PER_DISH: int = 4


def generate_instance(n_dishes: int, tightness: str, seed: int) -> list[Dish]:
    """Create one random kitchen instance.

    Args:
        n_dishes: how many dishes to create (ids are 1..n_dishes).
        tightness: "tight", "moderate" or "loose".
        seed: random seed; the same seed always gives the same dishes.
    """
    if tightness not in SLACK_RANGES:
        raise ValueError(f"unknown tightness {tightness!r}")

    rng = random.Random(seed)
    slack_low, slack_high = SLACK_RANGES[tightness]
    last_arrival = ARRIVAL_MINUTES_PER_DISH * n_dishes

    dishes: list[Dish] = []
    for dish_id in range(1, n_dishes + 1):
        station = rng.choice(STATIONS)
        arrival = rng.randint(0, last_arrival)
        duration = rng.randint(*DURATION_RANGE)
        slack = rng.randint(slack_low, slack_high)
        deadline = arrival + duration + slack
        dishes.append(Dish(dish_id, station, arrival, duration, deadline))
    return dishes


# Idle minutes a station may sit empty between two planted dishes.
# Smaller numbers mean a busier kitchen.
IDLE_GAP_RANGE: tuple[int, int] = (0, 3)


def generate_feasible_instance(
    n_dishes: int, tightness: str, seed: int
) -> tuple[list[Dish], dict[int, int]]:
    """Create one random instance that is GUARANTEED to have a solution.

    How it works ("planted solution"):
      1. On each station, lay dishes one after another with small idle gaps.
         This gives a start time for every dish with no overlap -- a valid
         schedule we know exists.
      2. Around each planted start time, choose the arrival and deadline so the
         dish's slack falls in the tightness range and the planted start is
         still allowed.
      3. Shuffle the dish ids, so the id order does not reveal the planted
         schedule (basic backtracking tries dishes in id order).

    Returns:
        (dishes, planted_schedule). planted_schedule maps dish_id -> start
        minute. It proves the instance is feasible; solvers never see it.
    """
    if tightness not in SLACK_RANGES:
        raise ValueError(f"unknown tightness {tightness!r}")

    rng = random.Random(seed)
    slack_low, slack_high = SLACK_RANGES[tightness]
    next_free = {station: 0 for station in STATIONS}

    # Step 1 + 2: build (station, arrival, duration, deadline, planted_start).
    rows = []
    for _ in range(n_dishes):
        station = rng.choice(STATIONS)
        duration = rng.randint(*DURATION_RANGE)
        start = next_free[station] + rng.randint(*IDLE_GAP_RANGE)
        next_free[station] = start + duration

        slack = rng.randint(slack_low, slack_high)
        before = min(rng.randint(0, slack), start)  # arrival cannot be < 0
        after = slack - before
        arrival = start - before
        deadline = start + duration + after
        rows.append((station, arrival, duration, deadline, start))

    # Step 3: shuffle, then number the dishes 1..n in the shuffled order.
    rng.shuffle(rows)
    dishes: list[Dish] = []
    planted: dict[int, int] = {}
    for dish_id, (station, arrival, duration, deadline, start) in enumerate(rows, 1):
        dishes.append(Dish(dish_id, station, arrival, duration, deadline))
        planted[dish_id] = start
    return dishes, planted
