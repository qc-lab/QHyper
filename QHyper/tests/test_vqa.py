import numpy as np
import pytest

from QHyper.solvers import solver_from_config

np.random.seed(1244)


def assert_probabilities(probabilities, expected_probabilities):
    for rec in probabilities:
        key = ''.join([str(rec[f'x{i}']) for i in range(5)])
        assert rec.probability == pytest.approx(expected_probabilities[key])


def get_problem_config():
    problem_config = {
        "type": "knapsack",
        "max_weight": 2,
        "item_weights": [1, 1, 1],
        "item_values": [2, 2, 1],
    }

    params_config = {
        'gamma': {
            'init': [0.5]*3,
        },
        'beta': {
            'init': [1]*3,
        },
        # penalty_weights': {
        # 'angles': [[0.5]*3, [1]*3],
        # 'hyper_args': [1, 2.5, 2.5],
    }
    hyper_optimizer_bounds = 3*[(1, 10)]

    return problem_config, params_config, hyper_optimizer_bounds


def run_solver(solver_config):
    vqa = solver_from_config(solver_config)
    return vqa.solve(None)


def test_qaoa():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            'name': 'QAOA',
            'category': 'gate_based',
            'platform': 'pennylane',
            'layers': 3,
            'penalty_weights': [1, 2.5, 2.5],
            **params_config
        },
        "problem": problem_config
    }

    solver_results = run_solver(solver_config)
    assert_probabilities(solver_results.probabilities, {
        '00000': 0.052147632,
        '00001': 0.047456206,
        '00010': 0.067478508,
        '00011': 0.120734632,
        '00100': 0.019935786,
        '00101': 0.005007856,
        '00110': 0.010055809,
        '00111': 0.022499649,
        '01000': 0.027125455,
        '01001': 0.028211751,
        '01010': 0.008350237,
        '01011': 0.032026917,
        '01100': 0.003062964,
        '01101': 0.012204283,
        '01110': 0.086633588,
        '01111': 0.003494550,
        '10000': 0.027125455,
        '10001': 0.028211751,
        '10010': 0.008350237,
        '10011': 0.032026917,
        '10100': 0.003062964,
        '10101': 0.012204283,
        '10110': 0.086633588,
        '10111': 0.003494550,
        '11000': 0.010496304,
        '11001': 0.015135842,
        '11010': 0.115724937,
        '11011': 0.010027077,
        '11100': 0.007232224,
        '11101': 0.013160777,
        '11110': 0.039326178,
        '11111': 0.041361077,
    })


def test_wfqaoa():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            'name': 'WF_QAOA',
            'category': 'gate_based',
            'platform': 'pennylane',
            "layers": 3,
            "limit_results": 10,
            "penalty": 2,
            "backend": "default.qubit",
            "penalty_weights": [1, 2.5, 2.5],
            **params_config,
        },
        "problem": problem_config
    }

    solver_results = run_solver(solver_config)
    assert_probabilities(solver_results.probabilities, {
        '00000': 0.052147632,
        '00001': 0.047456206,
        '00010': 0.067478508,
        '00011': 0.120734632,
        '00100': 0.019935786,
        '00101': 0.005007856,
        '00110': 0.010055809,
        '00111': 0.022499649,
        '01000': 0.027125455,
        '01001': 0.028211751,
        '01010': 0.008350237,
        '01011': 0.032026917,
        '01100': 0.003062964,
        '01101': 0.012204283,
        '01110': 0.086633588,
        '01111': 0.003494550,
        '10000': 0.027125455,
        '10001': 0.028211751,
        '10010': 0.008350237,
        '10011': 0.032026917,
        '10100': 0.003062964,
        '10101': 0.012204283,
        '10110': 0.086633588,
        '10111': 0.003494550,
        '11000': 0.010496304,
        '11001': 0.015135842,
        '11010': 0.115724937,
        '11011': 0.010027077,
        '11100': 0.007232224,
        '11101': 0.013160777,
        '11110': 0.039326178,
        '11111': 0.041361077,
    })


def test_hqaoa():
    problem_config, params_config, _ = get_problem_config()

    solver_config = {
        "solver": {
            "name": "H_QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": 3,
            "limit_results": 10,
            "penalty": 2,
            "backend": "default.qubit",
            **params_config,
            'penalty_weights': {
                'init': [1, 2.5, 2.5],
            }
        },
        "problem": problem_config
    }

    solver_results = run_solver(solver_config)
    assert_probabilities(solver_results.probabilities, {
        '00000': 0.052147632,
        '00001': 0.047456206,
        '00010': 0.067478508,
        '00011': 0.120734632,
        '00100': 0.019935786,
        '00101': 0.005007856,
        '00110': 0.010055809,
        '00111': 0.022499649,
        '01000': 0.027125455,
        '01001': 0.028211751,
        '01010': 0.008350237,
        '01011': 0.032026917,
        '01100': 0.003062964,
        '01101': 0.012204283,
        '01110': 0.086633588,
        '01111': 0.003494550,
        '10000': 0.027125455,
        '10001': 0.028211751,
        '10010': 0.008350237,
        '10011': 0.032026917,
        '10100': 0.003062964,
        '10101': 0.012204283,
        '10110': 0.086633588,
        '10111': 0.003494550,
        '11000': 0.010496304,
        '11001': 0.015135842,
        '11010': 0.115724937,
        '11011': 0.010027077,
        '11100': 0.007232224,
        '11101': 0.013160777,
        '11110': 0.039326178,
        '11111': 0.041361077,
    })
