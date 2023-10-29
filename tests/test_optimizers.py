import numpy as np
import pytest

from QHyper.solvers import solver_from_config
from QHyper.util import weighted_avg_evaluation

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
        results.results_probabilities, vqa.problem.get_score, 0)


def test_scipy():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qaoa",
                "layers": 5,
                "backend": "default.qubit",
            },
            "optimizer": {
                "type": "scipy",
                "maxfun": 10
            },
            "params_inits": params_config,
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.310672703)


def test_qml():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qaoa",
                "layers": 5,
                "backend": "default.qubit",
            },
            "optimizer": {
                "type": "qml",
                "optimization_steps": 10
            },
            "params_inits": params_config,
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.171165308)


def test_qml_qaoa():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qml_qaoa",
                "layers": 5,
                "backend": "default.qubit",
                "optimizer": "adam",
                "optimizer_args": {
                    "optimization_steps": 10
                }
            },
            "params_inits": params_config,
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.171165308)


def test_random():
    problem_config, params_config, hyperoptimizer_bounds = get_problem_config()

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qaoa",
                "layers": 5,
                "backend": "default.qubit",
            },
            "optimizer": {
                "type": "qml",
                "optimization_steps": 10
            },
            "hyper_optimizer": {
                "type": "random",
                "processes": 1,
                "number_of_samples": 2,
                "bounds": hyperoptimizer_bounds,
                "disable_tqdm": False
            },
            "params_inits": params_config,
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-1.536984353)


def test_cem():
    problem_config, params_config, hyperoptimizer_bounds = get_problem_config()

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qaoa",
                "layers": 5,
                "backend": "default.qubit",
            },
            "optimizer": {
                "type": "qml",
                "optimization_steps": 10
            },
            "hyper_optimizer": {
                "type": "cem",
                "processes": 1,
                "samples_per_epoch": 2,
                "epochs": 2,
                "bounds": hyperoptimizer_bounds,
                "disable_tqdm": False
            },
            "params_inits": params_config,
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.171165308)


def test_grid():
    problem_config, params_config, hyperoptimizer_bounds = get_problem_config()

    solver_config = {
        "solver": {
            "type": "vqa",
            "pqc": {
                "type": "qaoa",
                "layers": 5,
                "backend": "default.qubit",
            },
            "optimizer": {
                "type": "qml",
                "optimization_steps": 10
            },
            "hyper_optimizer": {
                "type": "grid",
                "processes": 1,
                "steps": [8, 7, 6],
                "bounds": hyperoptimizer_bounds,
                "disable_tqdm": False
            },
            "params_inits": params_config,
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-1.014492067)
