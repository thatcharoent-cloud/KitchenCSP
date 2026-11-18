"""Unit tests for the random instance generator."""

import pytest

from src.evaluation.validator import validate_schedule
from src.generation.generator import (
    SLACK_RANGES,
    generate_feasible_instance,
    generate_instance,
)

SIZES = [5, 10, 15, 20]
TIGHTNESS = ["tight", "moderate", "loose"]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("tightness", TIGHTNESS)
def test_random_instance_shape(n, tightness):
    dishes = generate_instance(n, tightness, seed=7)
    low, high = SLACK_RANGES[tightness]
    assert [d.id for d in dishes] == list(range(1, n + 1))
    assert all(low <= d.slack <= high for d in dishes)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("tightness", TIGHTNESS)
def test_feasible_instance_has_valid_planted_schedule(n, tightness):
    for seed in range(20):
        dishes, planted = generate_feasible_instance(n, tightness, seed)
        low, high = SLACK_RANGES[tightness]
        assert sorted(d.id for d in dishes) == list(range(1, n + 1))
        assert all(low <= d.slack <= high for d in dishes)
        assert validate_schedule(dishes, planted)


def test_same_seed_gives_same_instance():
    assert generate_instance(10, "tight", 3) == generate_instance(10, "tight", 3)
    assert generate_feasible_instance(10, "tight", 3) == generate_feasible_instance(
        10, "tight", 3
    )


def test_different_seeds_give_different_instances():
    assert generate_instance(10, "tight", 1) != generate_instance(10, "tight", 2)


def test_unknown_tightness_is_rejected():
    with pytest.raises(ValueError):
        generate_instance(5, "very tight", 0)
    with pytest.raises(ValueError):
        generate_feasible_instance(5, "very tight", 0)
