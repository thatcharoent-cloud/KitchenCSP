"""Dish data model for KitchenCSP.

A dish is a single non-preemptive preparation job that requires one
single-capacity station. Time is measured in integer minutes.

Convention: dish IDs are plain integers 1..n and are
the deterministic tie-breaker whenever two dishes compare equal on the
primary sort key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Station types, each with capacity one. Extending the kitchen model later
# only means adding names here (and updating generator/validator tests).
Station = Literal["wok", "fryer", "curry"]
STATIONS: tuple[Station, ...] = ("wok", "fryer", "curry")


def _is_int(value: object) -> bool:
    """True for plain ints. Python's bool is an int subclass, so reject bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def ensure_unique_ids(dishes: list[Dish], context: str) -> None:
    """Raise ValueError if any dish id appears more than once in `dishes`.

    Solvers dispatch on dish ids (e.g. `schedule[dish.id] = start`), so a
    duplicate id would silently corrupt the schedule. Every solver and the
    validator call this guard.
    """
    seen: set[int] = set()
    duplicates: set[int] = set()
    for dish in dishes:
        if dish.id in seen:
            duplicates.add(dish.id)
        seen.add(dish.id)
    if duplicates:
        raise ValueError(f"{context}: duplicate dish id(s): {sorted(duplicates)}")


@dataclass(frozen=True)
class Dish:
    """An immutable dish/job specification.

    Attributes:
        id: Unique integer identifier (deterministic tie-breaker).
        station: Which single-capacity station this dish requires.
        arrival: Earliest minute the dish may start (inclusive).
        duration: Required cooking minutes; the dish is non-preemptive.
        deadline: Latest minute the dish may finish (inclusive).
    """

    id: int
    station: Station
    arrival: int
    duration: int
    deadline: int

    def __post_init__(self) -> None:
        # Python's `True` is an int, so every integer field must also reject
        # bool values (see _is_int above).
        if not _is_int(self.id) or self.id < 1:
            raise ValueError("id must be a positive integer (dish ids are 1..n)")
        if self.station not in STATIONS:
            raise ValueError(
                f"unknown station {self.station!r}; expected one of {STATIONS}"
            )
        if not _is_int(self.arrival) or self.arrival < 0:
            raise ValueError("arrival must be a non-negative integer")
        if not _is_int(self.duration) or self.duration <= 0:
            raise ValueError("duration must be a positive integer")
        if not _is_int(self.deadline) or self.deadline < self.arrival:
            raise ValueError("deadline must be an integer >= arrival")

        # Negative slack is deliberately ALLOWED here: it means the initial
        # domain is empty, so the instance is infeasible before any search.
        # Edge-case tests rely on being able to construct one.

    @property
    def slack(self) -> int:
        """Schedule slack S_i = deadline - arrival - duration (minutes)."""
        return self.deadline - self.arrival - self.duration

    @property
    def latest_start(self) -> int:
        """Latest valid start minute, i.e. deadline - duration."""
        return self.deadline - self.duration

    @property
    def domain(self) -> range:
        """Initial domain of valid start minutes: arrival .. deadline - duration.

        Returns an empty range when the dish is infeasible on its own
        (negative slack). Solvers copy working domains (sets/lists) from this
        immutable blueprint; the blueprint itself never changes.
        """
        return range(self.arrival, self.latest_start + 1)
