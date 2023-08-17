import numpy as np
import argparse

from QHyper.problems import KnapsackProblem, TSPProblem
from QHyper.solvers import Solver


np.random.seed(1244)


def get_problem(problem_name):
    if problem_name == 'knapsack':
        problem = KnapsackProblem(max_weight=2, items=[(1, 2), (1, 2),(1, 1)])
        params_cofing = {
            'angles': [[0.5]*5, [1]*5],
            'hyper_args': [1, 2.5, 2.5],
        }
        hyper_optimizer_bounds = 3*[(1, 10)]

    elif problem_name == 'tsp':
        problem = TSPProblem(
            number_of_cities=3,
        )
        params_cofing = {
            'angles': [[0.5]*5, [1]*5],
            'hyper_args': [1, 2, 2, 2, 2],
        }
        hyper_optimizer_bounds = 5*[(1, 10)]
    
    return problem, params_cofing, hyper_optimizer_bounds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", type=str, help="Problem name")
    args = parser.parse_args()

    problem, params_cofing, hyper_optimizer_bounds = get_problem(args.problem)

    print(f"Variables used to describe objective function"
        f"and constraints: {problem.variables}")
    print(f"Objective function: {problem.objective_function}")
    print("Constraints:")
    for constraint in problem.constraints:
        print(f"    {constraint}")
    
    tester_config = {
        "solver": {
            "type": "vqa",
            "args": {
                "config": {
                    "pqc": {
                        "type": "wfqaoa",
                        "layers": 5,
                    }
                }
            }
        },
    }
    tester = Solver.from_config(problem, tester_config)

    solver_config = {
        "solver": {
            "type": "vqa",
            "args": {
                "config": {
                    "optimizer": {
                        "type": "scipy",
                        "maxfun": 10,
                    },
                    "pqc": {
                        "type": "wfqaoa",
                        "layers": 5,
                    }
                }
            }
        },
        "hyper_optimizer": {
            "type": "random",    
            "processes": 2,
            "number_of_samples": 4,
            "disable_tqdm": True,
            "bounds": hyper_optimizer_bounds
        }
    }
    vqa = Solver.from_config(problem, solver_config)
    
    best_params = vqa.solve(params_cofing)
    print(f"Best params: {best_params}")
    best_results = tester.evaluate(best_params)
    print(f"Best results: {best_results}")
    print(f"Params used for optimizer:\n{best_params['angles']},\n"
        f"and params used for hyper optimizer: {best_params['hyper_args']}")
