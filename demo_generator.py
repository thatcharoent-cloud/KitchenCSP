from src.generation.generator import generate_instance, generate_feasible_instance
from src.algorithms.greedy import fcfs, edf
from src.algorithms.backtracking import backtracking
from src.evaluation.validator import validate_schedule


def run_all(dishes):
    for r in (fcfs(dishes), edf(dishes), backtracking(dishes)):
        print(" ", r.algorithm, r.status.value, r.schedule)


print("=== Random instance (may have no solution) ===")
dishes = generate_instance(n_dishes=5, tightness="tight", seed=1)
for d in dishes:
    print(" ", d)
run_all(dishes)

print()
print("=== Feasible instance (solution guaranteed) ===")
dishes, planted = generate_feasible_instance(n_dishes=5, tightness="tight", seed=1)
for d in dishes:
    print(" ", d)
print("  planted solution:", planted, "valid =", validate_schedule(dishes, planted))
run_all(dishes)
