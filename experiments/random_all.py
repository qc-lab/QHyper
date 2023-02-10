import pennylane as qml
import numpy as np
import multiprocessing as mp
import itertools

from QHyper.optimizers.cem import CEM
from QHyper.optimizers.random import Random
from QHyper.optimizers.shgo import Shgo
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.solvers.qaoa.core import QAOA
from QHyper.optimizers.scipy_minimizer import ScipyOptimizer
from QHyper.optimizers.scipy_all_minimizer import ScipyAllOptimizer

from QHyper.problems.knapsack import Knapsack, KnapsackProblem


KNAPSACK_PARAMS = {
    'max_weight': 2,
    'items': [(1, 2), (1, 2), (1, 1)]
}
PROCESSES = 5


def compute(params):
    angles, weights = params
    knapsack_qaoa = KnapsackProblem(**KNAPSACK_PARAMS)

    solver = QAOA(
        problem=knapsack_qaoa,
        platform="pennylane",
        hyperoptimizer=ScipyAllOptimizer(maxfun=200, bounds=[(1, 10)]*2 + [(0, 2*np.pi)]*10),
        layers=5,
        angles=angles,
        weights=weights,
    )

    value, params, weights = solver.solve()
    return value, params, weights


def run_one(samples):
    angles = 2 * np.pi * np.random.rand(samples, 2, 5)
    weights = 10*np.random.rand(samples, 2) + 1

    with mp.Pool(processes=PROCESSES) as p:
        results = list(p.imap(compute, zip(angles, weights)))

    min_idx = np.argmin([result[0] for result in results])
    return results[min_idx][0]


if __name__ == '__main__':
    print(f"Random all {KNAPSACK_PARAMS}")
    # tests = [(500, 1), (500, 2), (250, 1), (250, 2), (250, 4), (100, 1), (100, 2), (100, 5), (100, 10)]
    tests = [1000, 500, 250, 100]
    for samples in tests:
        results = [run_one(samples) for _ in range(10)]
        print(f"samples: {samples}, mean: {np.mean(results)}, std: {np.std(results)}")
