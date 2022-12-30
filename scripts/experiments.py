import pennylane as qml
import numpy as np
import yaml

from QHyper.optimizers.cem import CEM
from QHyper.optimizers.random import Random
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.solvers.qaoa.core import QAOA

from QHyper.problems.tsp import TSPProblem
from QHyper.problems.knapsack import KnapsackProblem


def run_experiments(problems, hyperoptimizers, angles_alpha, angles_beta):
    for angle_alpha in angles_alpha:
        for angle_beta in angles_beta:
            for problem in problems:
                for hyperopt in hyperoptimizers:    
                    print(f"{problem} {hyperopt} {angle_alpha} {angle_beta}")
                    solver = QAOA(
                        problem=problem,
                        platform="pennylane",
                        optimizer=QmlGradientDescent(qml.AdamOptimizer(stepsize=0.05), 200),
                        layers=5,
                        angles=[[angle_alpha]*5, [angle_beta]*5],
                        weights=[1, 1, 1],
                        hyperoptimizer=hyperopt
                        # backend=
                    )
                    value, _, _ = solver.solve()
                    print(value)



def create_instances(config, group):
    result = []

    classes = {
        'problems': {
            'knapsack': KnapsackProblem,
            'tsp': TSPProblem
        },
        'hyperoptimizers': {
            'CEM': CEM,
            'Random': Random,
            'None': None
        }
    }

    for k, v in config[group].items():
        class_name, name = x if len(x:=k.split('_')) == 2 else (x[0], "")
        if class_name not in classes[group]:
            raise Exception(f"Cannot find {class_name} in {classes[group].keys()}")
        if class_name == 'None':
            result.append(None)
            continue
        
        instance = classes[group][class_name](**v)
        instance.name = name
        result.append(instance)
    return result

if __name__ == '__main__':
    with open('experiments.yaml') as file:
        config = yaml.safe_load(file)
    print(config['problems'])
    problems = create_instances(config, 'problems')
    hyperoptimizers = create_instances(config, 'hyperoptimizers')
    print(problems)
    print(hyperoptimizers)

    run_experiments(
        problems, hyperoptimizers, config['angles_alpha'], config['angles_beta'])
