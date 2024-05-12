import numpy as np
import pytest

from QHyper.solvers import solver_from_config
from QHyper.util import weighted_avg_evaluation
from QHyper.optimizers import (
    ScipyOptimizer, OptimizationResult, QmlGradientDescent,
    Random, GridSearch, CEM
)

np.random.seed(1244)


def get_problem_config():
    problem_config = {
        "type": "knapsack",
        "max_weight": 2,
        "items": [(1, 2), (1, 2), (1, 1)]
    }

    params_config = {
        'angles': [[0.5]*5, [1]*5],
        'hyper_args': [1, 2.5, 2.5],
    }
    hyper_optimizer_bounds = 3*[(1, 10)]

    return problem_config, params_config, hyper_optimizer_bounds


def run_solver(solver_config):
    vqa = solver_from_config(solver_config)
    results = vqa.solve()
    return weighted_avg_evaluation(
        results.probabilities, vqa.problem.get_score, 0)


def function(args) -> OptimizationResult:
    x, y, z = args
    return OptimizationResult((x + y + z)**2, np.array([x, y, z]), [[]])


def test_scipy():
    minimizer = ScipyOptimizer(
        maxfun=100,
        bounds=np.array([(-1, 1), (-1, 1), (-1, 1)]),
        method='L-BFGS-B'
    )
    result = minimizer.minimize(function, np.array([1., 0.5, -0.3]))
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_qml():
    minimizer = QmlGradientDescent(
        # bounds=[(-1, 1), (-1, 1), (-1, 1)],
    )
    result = minimizer.minimize(function, np.array([1., 0.5, -0.3]))
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_random():
    minimizer = Random(
        processes=1,
        number_of_samples=1000,
        bounds=np.array([(-1, 1), (-1, 1), (-1, 1)]),
        disable_tqdm=True,
    )
    result = minimizer.minimize(function, np.array([1., 0.5, -0.3]))
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_cem():
    minimizer = CEM(
        processes=1,
        samples_per_epoch=100,
        epochs=3,
        bounds=np.array([(-1, 1), (-1, 1), (-1, 1)]),
        disable_tqdm=True,
    )
    result = minimizer.minimize(function, np.array([1., 0.5, -0.3]))
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_grid():
    minimizer = GridSearch(
        processes=1,
        steps=[0.5, 0.5, 0.5],
        bounds=np.array([(-1, 1), (-1, 1), (-1, 1)]),
        disable_tqdm=True,
    )
    result = minimizer.minimize(function, np.array([1., 0.5, -0.3]))
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)
