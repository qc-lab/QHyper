import numpy as np
import pytest
import random

from QHyper.solvers import solver_from_config
from QHyper.util import weighted_avg_evaluation


def get_problem_config():
    problem_config = {
        "type": "knapsack",
        "max_weight": 2,
        "item_weights": [1, 1, 1],
        "item_values": [2, 2, 1],
    }

    params_config = {
        'gamma': {
            'init': [0.5]*5,
            'min': [0]*5,
            'max': [2*np.pi]*5,
        },
        'beta': {
            'init': [1]*5,
            'min': [0]*5,
            'max': [2*np.pi]*5,
        },
        # 'angles': [[0.5]*5, [1]*5],
        # 'hyper_args': [1, 2.5, 2.5],
    }
    hyper_optimizer_bounds = 3*[(1, 10)]

    return problem_config, params_config, hyper_optimizer_bounds


def run_solver(solver_config):
    np.random.seed(0)
    random.seed(0)

    solver = solver_from_config(solver_config)
    results = solver.solve()
    return weighted_avg_evaluation(
        results.probabilities, solver.problem.get_score, 0)


def run_hyper_optimizer(solver_config):
    np.random.seed(0)
    random.seed(0)

    solver = solver_from_config(solver_config)
    results = solver.solve()
    return results.value


def test_scipy():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "name": "QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 5,
            **params_config,
            "optimizer": {
                "type": "scipy",
                "maxfun": 10,
                # "bounds": [(0, 2*np.pi)]*10,
                'method': 'L-BFGS-B',
                'options': {
                    'maxiter': 10,
                }
            },
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.4697774822)


def test_qml():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "name": "QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 5,
            **params_config,
            "optimizer": {
                "type": "qml",
                "steps": 10
            },
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.4015307189)


def test_qml_qaoa():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "name": "QML_QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 5,
            "backend": "default.qubit",
            **params_config,
            "optimizer": {
                "type": "qml",
                "name": "adam",
                "steps": 10
            },
        },
        "problem": problem_config
    }

    result = run_solver(solver_config)
    assert result == pytest.approx(-0.4015307189)


def test_random():
    problem_config, params_config, hyperoptimizer_bounds = get_problem_config()

    solver_config = {
        "solver": {
            "name": "QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 5,
            **params_config,
            # "type": "vqa",
            # "pqc": {
            #     "type": "qaoa",
            #     "layers": 5,
            #     "backend": "default.qubit",
            # },
            "optimizer": {
                "type": "qml",
                "steps": 10
            },

            # "params_inits": params_config,
        },
        "hyper_optimizer": {
            "optimizer": {
                "type": "random",
                "processes": 1,
                "number_of_samples": 2,
                "disable_tqdm": False
            },
            'penalty_weights': {
                'min': [1]*3,
                'max': [10]*3,
            },
        },
        "problem": problem_config
    }

    result = run_hyper_optimizer(solver_config)
    assert result == pytest.approx(-0.5250361108, rel=1e-6, abs=1e-6)


def test_cem():
    problem_config, params_config, hyperoptimizer_bounds = get_problem_config()

    solver_config = {
        "solver": {
            "name": "QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 5,
            **params_config,
            "optimizer": {
                "type": "qml",
                "steps": 10
            },
        },
        "hyper_optimizer": {
            "optimizer": {
                "type": "cem",
                "processes": 1,
                "samples_per_epoch": 2,
                "epochs": 2,
                "disable_tqdm": False,
            },
            'penalty_weights': {
                'min': [1]*3,
                'max': [10]*3,
                'init': [1, 2.5, 2.5],
            },
        },
        "problem": problem_config
    }

    result = run_hyper_optimizer(solver_config)
    assert result == pytest.approx(-0.5078221819, rel=1e-6, abs=1e-6)


def test_grid():
    problem_config, params_config, hyperoptimizer_bounds = get_problem_config()

    solver_config = {
        "solver": {
            "name": "QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 5,
            **params_config,
            "optimizer": {
                "type": "qml",
                "steps": 10
            },
        },
        "hyper_optimizer": {
            "optimizer": {
                "type": "grid",
                "processes": 1,
                "disable_tqdm": False,
            },
            'penalty_weights': {
                'min': [1]*3,
                'max': [10]*3,
                'step': [8, 7, 6],
            },
        },
        "problem": problem_config
    }

    result = run_hyper_optimizer(solver_config)
    assert result == pytest.approx(-1.014492067)
