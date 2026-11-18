# KitchenCSP

**Research title:** Under What Conditions Does Constraint Satisfaction Outperform Greedy Scheduling in Restaurant Kitchen Order Allocation?

## Overview

KitchenCSP studies whether constraint satisfaction can find feasible restaurant kitchen schedules more reliably than simple greedy methods, and how much additional computation it requires. The focus is on a small, reproducible experiment rather than a restaurant management application.

A dish is a scheduling job with an arrival time, preparation duration, deadline, and required station. The initial kitchen has three single-capacity stations: wok, fryer, and curry. Cooking cannot be interrupted.

## Research question

How do order volume, deadline tightness, and competition for kitchen stations affect the feasibility and runtime of constraint satisfaction compared with greedy scheduling?

## CSP formulation

- **Variables:** Integer start time `X_i` for each dish `i`.
- **Domains:** All integer minutes from arrival `a_i` through `d_i - p_i`, inclusive, where `p_i` is cooking duration and `d_i` is the completion deadline.
- **Constraints:** Each dish starts no earlier than its arrival and finishes by its deadline; dishes assigned to the same station must not overlap.
- **Objective:** Find any feasible schedule. Minimizing lateness is outside the initial scope.

## Algorithms to compare

1. **FCFS:** Process dishes in arrival order and place each at its earliest available valid time, without revising previous placements.
2. **EDF:** Process dishes in deadline order and place each at its earliest available valid time, without revising previous placements.
3. **Basic backtracking:** Exhaustively search valid start times using a fixed variable order.
4. **Backtracking + MRV + forward checking:** Select the smallest remaining domain and prune conflicting start times after each assignment.

## Proposed experiments

Use identical synthetic instances for every method, with fixed random seeds. Initially test 5, 10, 15, and 20 dishes, with 20 instances for each combination of dish count and deadline tightness (240 total). Define slack as `deadline - arrival - duration`:

- Tight: 0–3 minutes
- Moderate: 4–8 minutes
- Loose: 9–15 minutes

These are experimental settings, not industry standards. Include some constructed feasible instances, and separately record instances proved infeasible by exhaustive search. Consider station congestion as a follow-up experiment if time permits.

## Metrics

- Feasible schedule discovery rate, including the rate on instances known to be feasible
- Runtime in milliseconds
- Attempted assignments and explored nodes for CSP methods
- Independent validation of every returned schedule

A timeout means **unknown**, not infeasible. Greedy failure means only that its chosen ordering did not produce a feasible schedule.

## Scope and limitations

This initial model uses fixed durations, known arrivals and deadlines, one station per dish, and no interruptions. It does not model staffing, uncertain preparation times, live incoming orders, multi-dish synchronization, or optimal tardiness.

## Development and reproducibility

This repository contains Python implementations, unit tests, scenario generation, experimental results, and a reproducible run guide. Exact commands and dependencies are documented as each component is implemented and verified. No experimental results should be claimed before they are measured.

### How to run

Requires Python 3.10 or newer (tested on 3.13). Run every command from the project root.

```bash
python3 -m pip install -r requirements.txt   # installs pytest
python3 -m pytest                            # runs all unit tests
python3 -m src.generation.suite              # generates the instance suites
```

The last command writes two files:

- `data/instances_random.json`: 240 random instances (some may have no solution).
- `data/instances_feasible.json`: 240 instances built around a planted valid schedule, so a solution is guaranteed. The planted schedule is stored with each instance but is never given to the solvers.

Each instance has a fixed seed, `n_dishes * 10000 + tightness_index * 1000 + index` (tightness index 0 = tight, 1 = moderate, 2 = loose), so the files are identical on every machine. They are generated rather than committed: re-run the command to recreate them.

## Research integrity and reproduction

All results in this repository are measured, not assumed. Instances are generated with fixed seeds and their parameters recorded; every returned schedule is checked by an independent validator; greedy failure, proven infeasibility, and timeout are recorded separately and never conflated. Any code or data reused from other projects is cited transparently. Limitations that affect reproducibility are documented rather than hidden.

