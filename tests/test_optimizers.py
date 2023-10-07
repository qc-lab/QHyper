import numpy as np
import pytest

from QHyper.problems import KnapsackProblem
from QHyper.solvers import Solver
from QHyper.util import weighted_avg_evaluation

np.random.seed(1244)


def get_problem():
    problem = KnapsackProblem(max_weight=2, items=[(1, 2), (1, 2),(1, 1)])
    params_cofing = {
        'angles': [[0.5]*5, [1]*5],
        'hyper_args': [1, 2.5, 2.5],
    }
    hyper_optimizer_bounds = 3*[(1, 10)]
 
    return problem, params_cofing, hyper_optimizer_bounds

def run_solver(problem, solver_config, params_config):
    vqa = Solver.from_config(problem, solver_config) 
    results = vqa.solve(params_config)
    return weighted_avg_evaluation(
        results.results_probabilities, problem.get_score, 0)


def test_scipy():
    problem, params_cofing, hyper_optimizer_bounds = get_problem()

    solver_config = {
        "solver": {
            "type": "vqa",
            "args": {
                "config": {
                    "pqc": {
                        "type": "qaoa",
                        "layers": 5,
                        "backend": "default.qubit",
                    },
                    "optimizer": {
                        "type": "scipy",
                        "maxfun": 10
                    }
                }
            }
        }
    }

    result = run_solver(problem, solver_config, params_cofing)
    assert result == pytest.approx(-0.310672703)


def test_qml():
    problem, params_cofing, hyper_optimizer_bounds = get_problem()

    solver_config = {
        "solver": {
            "type": "vqa",
            "args": {
                "config": {
                    "pqc": {
                        "type": "qaoa",
                        "layers": 5,
                        "backend": "default.qubit",
                    },
                    "optimizer": {
                        "type": "qml",
                        "optimization_steps": 10
                    }
                }
            }
        }
    }

    result = run_solver(problem, solver_config, params_cofing)
    assert result == pytest.approx(-0.171165308)


def test_random():
    problem, params_cofing, hyper_optimizer_bounds = get_problem()

    solver_config = {
        "solver": {
            "type": "vqa",
            "args": {
                "config": {
                    "pqc": {
                        "type": "qaoa",
                        "layers": 5,
                        "backend": "default.qubit",
                    },
                    "optimizer": {
                        "type": "qml",
                        "optimization_steps": 10
                    }
                }
            }
        },
        "hyper_optimizer": {
            "type": "random",    
            "processes": 1,
            "number_of_samples": 2,
            "bounds": hyper_optimizer_bounds,
            "disable_tqdm": False
        }
    }

    result = run_solver(problem, solver_config, params_cofing)
    assert result == pytest.approx(-1.536984353)


def test_cem():
    problem, params_cofing, hyper_optimizer_bounds = get_problem()

    solver_config = {
        "solver": {
            "type": "vqa",
            "args": {
                "config": {
                    "pqc": {
                        "type": "qaoa",
                        "layers": 5,
                        "backend": "default.qubit",
                    },
                    "optimizer": {
                        "type": "qml",
                        "optimization_steps": 10
                    }
                }
            }
        },
        "hyper_optimizer": {
            "type": "cem",    
            "processes": 1,
            "samples_per_epoch": 2,
            "epochs": 2,
            "bounds": hyper_optimizer_bounds,
            "disable_tqdm": False
        }
    }

    result = run_solver(problem, solver_config, params_cofing)
    assert result == pytest.approx(-0.171165308)
