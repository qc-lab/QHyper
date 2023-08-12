import yaml
import numpy as np

from QHyper.problems.knapsack import KnapsackProblem
from QHyper.problems.tsp import TSPProblem
from QHyper.solvers.vqa.base import VQA
from QHyper.solvers import Solver

import pennylane as qml


if __name__ == '__main__':
    problem = KnapsackProblem(max_weight=4, items=[(2,1), (2, 2),(1, 1), (1, 3)])
    print(problem.knapsack.items)

    LAYERS = 5
   

    tester_config = {
        'pqc': {
            'type': 'wfqaoa',
            'layers': LAYERS,
        }
    }

    tester = VQA(problem, config=tester_config)

    # solver_config = {
    #     'optimizer': {
    #         'type': 'scipy',
    #         'maxfun': 200,
    #     },
    #     'pqc': {
    #         'type': 'wfqaoa',
    #         'layers': 5,
    #     },
    # }

    # read from yaml config file
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        print(f"Config: {config}")  
    vqa = Solver.from_config(problem, config)
    best_result = 0
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
    best_params = {
        'angles': [[1, 0.8, 0.6, 0.4, 0.2], [1]*LAYERS],
        'hyper_args': [1, 5, 5],
    }
    print(f"Best results: {best_result}")
    print(f"Params used for optimizer:\n{best_params['angles']},\n"
        f"and params used for hyper optimizer: {best_params['hyper_args']}")
    params = vqa.solve(best_params)
    results = tester.evaluate(params, True)
    print(f"Results: {results}")
    print(f"Best params: {params}")
