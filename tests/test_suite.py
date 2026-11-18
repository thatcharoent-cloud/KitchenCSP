"""Unit tests for building, saving and loading the experiment suite."""

import pytest

from src.evaluation.validator import validate_schedule
from src.generation.suite import (
    INSTANCES_PER_CELL,
    SIZES,
    TIGHTNESS,
    generate_suite,
    instance_seed,
    load_suite,
    save_suite,
)


@pytest.mark.parametrize("mode", ["random", "feasible"])
def test_suite_has_240_unique_instances(mode):
    records = generate_suite(mode)
    assert len(records) == len(SIZES) * len(TIGHTNESS) * INSTANCES_PER_CELL == 240
    assert len({r["instance_id"] for r in records}) == 240
    assert len({r["seed"] for r in records}) == 240


@pytest.mark.parametrize("mode", ["random", "feasible"])
def test_each_cell_has_20_instances_of_the_right_size(mode):
    records = generate_suite(mode)
    for n in SIZES:
        for tightness in TIGHTNESS:
            cell = [r for r in records if r["n_dishes"] == n and r["tightness"] == tightness]
            assert len(cell) == INSTANCES_PER_CELL
            assert all(len(r["dishes"]) == n for r in cell)


def test_feasible_suite_planted_schedules_are_valid():
    for r in generate_suite("feasible"):
        assert validate_schedule(r["dishes"], r["planted_schedule"])


def test_random_suite_has_no_planted_schedule():
    assert all(r["planted_schedule"] is None for r in generate_suite("random"))


def test_seed_is_readable():
    assert instance_seed(15, "moderate", 7) == 151007


def test_suite_is_reproducible():
    assert generate_suite("feasible") == generate_suite("feasible")


@pytest.mark.parametrize("mode", ["random", "feasible"])
def test_save_then_load_round_trip(tmp_path, mode):
    records = generate_suite(mode)
    path = tmp_path / f"{mode}.json"
    save_suite(records, path)
    assert load_suite(path) == records


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        generate_suite("easy")
