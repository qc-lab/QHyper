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

from QHyper.problems.knapsack import Knapsack, KnapsackProblem


KNAPSACK_PARAMS = {
    'max_weight': 2,
    'items': [(1, 2), (1, 2), (1, 1)]
}
PROCESSES = 5


def compute(angles, weights, samples):

    knapsack_qaoa = KnapsackProblem(**KNAPSACK_PARAMS)

    solver = QAOA(
        problem=knapsack_qaoa,
        platform="pennylane",
        optimizer=ScipyOptimizer(maxiter=200),
        layers=5,
        angles=angles,
        weights=weights,
        hyperoptimizer=Random(number_of_samples=samples, processes=PROCESSES, disable_tqdm=True, bounds=[(1, 10)]*2)
    )

    value, params, weights = solver.solve()
    return value


def run_one(angles_samples, weights_samples):
    results = []
    for _ in range(angles_samples):
        angles = 2 * np.pi *np.random.rand(10).reshape(2, 5)
        weights = 10*np.random.rand(2) + 1

        results.append(compute(angles, weights, weights_samples))

    return np.min(results)


if __name__ == '__main__':
    print(f"Random weights {KNAPSACK_PARAMS}")
    tests = [(1000, 1), (500, 1), (500, 2), (250, 1), (250, 2), (250, 4), (100, 1), (100, 2), (100, 5), (100, 10)]
    # tests = [1000, 500, 250, 100]
    for weights_samples, angles_samples in tests:
        results = [run_one(angles_samples, weights_samples) for _ in range(10)]
        print(f"weights_samples: {weights_samples}, angles_samples: {angles_samples}, mean: {np.mean(results)}, std: {np.std(results)}")
