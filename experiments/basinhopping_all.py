import pennylane as qml
import numpy as np

import multiprocessing as mp
import tqdm

from QHyper.optimizers.cem import CEM
from QHyper.optimizers.random import Random
from QHyper.optimizers.shgo import Shgo
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.solvers.qaoa.core import QAOA
from QHyper.optimizers.basinhopping import Basinhopping

from QHyper.problems.knapsack import Knapsack, KnapsackProblem

KNAPSACK_PARAMS = {
    'max_weight': 2,
    'items': [(1, 2), (1, 2), (1, 1)]
}
PROCESSES = 32

def wrapper(niter):
    # niter, maxiter = params
    knapsack_qaoa = KnapsackProblem(**KNAPSACK_PARAMS)
    angles = 2 * np.pi * np.random.rand(2, 5)
    weights = 10*np.random.rand(2) + 1
    solver = QAOA(
        problem=knapsack_qaoa,
        platform="pennylane",
        layers=5,
        angles=angles,
        weights=weights,
        hyperoptimizer=Basinhopping(niter=niter, maxiter=200, bounds=[(1, 10)]*2 + [(0, 2*np.pi)]*10),
    )
    result = solver.solve()[0]
    print(f"{niter},{200},{solver.counter},{result}")
    
    return result


if __name__ == '__main__':
    print(f"Basinhopping all {KNAPSACK_PARAMS}")
    # params = [
    #     (50, 100), (25, 100), (25, 50), (10, 50), (10, 100), (25, 25),
    #     (100, 50), (100, 25), (50, 25), (50, 10), (100, 10)] * 10
    params = [1000, 500, 250, 100, 50, 10] * 10
     
    with mp.Pool(processes=PROCESSES) as p:
        results = list(tqdm.tqdm(
            p.imap(wrapper, params), total=len(params), disable=True))
