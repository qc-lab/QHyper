import yaml
import numpy as np
import pennylane as qml

from QHyper.problems.knapsack import KnapsackProblem
from QHyper.problems.tsp import TSPProblem
from QHyper.solvers.vqa.base import VQA
from QHyper.solvers import Solver
from QHyper.util import (
    weighted_avg_evaluation, sort_solver_results, add_evaluation_to_results)


if __name__ == '__main__':
    problem = KnapsackProblem(max_weight=4, items=[(2,1), (2, 2),(1, 1), (1, 3)])
    print(problem.knapsack.items)

    LAYERS = 5

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        print(f"Config: {config}")
    vqa = Solver.from_config(problem, config)
    # best_result = 0
    # for i in range(200):
    #     params_cofing = {
    #         'angles': np.random.uniform(0, np.pi/2, (LAYERS, 2)),
    #         'hyper_args': [1, 5, 5],
    #     }
    #     # params_cofing = {
    #     #     'angles': [[1, 0.8, 0.6, 0.4, 0.2], [1]*LAYERS],
    #     #     'hyper_args': [1, 5, 5],
    #     # }
    #     # hyper_optimizer_bounds = 3*[(1, 10)]

    #     # result_params = vqa.solve(params_cofing)
    #     # print(f"Best params: {best_params}")

    #     results = tester.evaluate(params_cofing)
    #     if results < best_result:
    #         best_result = results
    #         best_params = params_cofing
    params_config = {
        'angles': [[1, 0.8, 0.6, 0.4, 0.2], [1]*LAYERS],
        'hyper_args': [1, 5, 5],
    }

    for _ in range(10):
        solver_results = vqa.solve(params_config)

    print("Evaluation:")
    print(weighted_avg_evaluation(
        solver_results.probabilities, problem.get_score,
        penalty=0, limit_results=10, normalize=True
    ))
    print("Sort results:")
    sorted_results = sort_solver_results(
        solver_results.probabilities, limit_results=10)

    results_with_evaluation = add_evaluation_to_results(
        sorted_results, problem.get_score, penalty=1)

    for result, (probability, evaluation) in results_with_evaluation.items():
        print(f"Result: {result}, "
            f"Prob: {probability:.5}, "
            f"Evaluation: {evaluation}")
        print("Solver results:")
