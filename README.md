# <img width="40" alt="qhyper_logo" src="docs/source/_static/logo.png" class="center"> QHyper
A software framework for hybrid quantum-classical optimization.

# Introduction

QHyper is a library, which is aimed at researchers working on
computational experiments with a variety of quantum combinatorial optimization solvers.
The library offers a simple and extensible interface for formulating combinatorial
optimization problems, selecting and running solvers,
and optimizing hyperparameters. The supported solver set includes variational
gate-based algorithms, quantum annealers, and classical solutions.
The solvers can be combined with provided local and global (hyper)optimizers.
The main features of the library are its extensibility on different levels of use
as well as a straightforward and flexible experiment configuration format.

# Documentation

Documentation can be found at [qhyper.readthedocs.io](https://qhyper.readthedocs.io/en/latest/)

# Code example

```python
from QHyper.solvers import solver_from_config

solver_config = {
    'solver': {
        'type': 'vqa',
            'optimizer': {
            'type': 'scipy',
            'maxfun': 200,
            "bounds": [(0, 2 * 3.1415)] * 10
        },
        'pqc': {
            'type': 'wfqaoa',
            'layers': 5,
        },
        'hyper_optimizer': {
            'type': 'random',
            'processes': 4,
            'number_of_samples': 100,
            'bounds': [(1, 10), (1, 10), (1, 10)]
        },
        'params_inits': {
            'angles': [[0.5]*5, [1]*5],
            'hyper_args': [1, 2.5, 2.5],
        }
    },
    'problem': {
        'type': 'knapsack',
        'max_weight': 2,
        'items_weights': [1, 1, 1],
        'items_values': [2, 2, 1],
    }
}
solver = solver_from_config(solver_config)
solver_results = solver.solve()

print(f"Probabilities: {solver_results.probabilities}")
# Probabilities: [(0, 0, 0, 0, 0, 0.00812924) (0, 0, 0, 0, 1, 0.01734155)
#                 (0, 0, 0, 1, 0, 0.01178566) (0, 0, 0, 1, 1, 0.00071086)
#                 (0, 0, 1, 0, 0, 0.00845063) (0, 0, 1, 0, 1, 0.00710329)
#                 (0, 0, 1, 1, 0, 0.03270284) (0, 0, 1, 1, 1, 0.02002345)
#                 (0, 1, 0, 0, 0, 0.01483474) (0, 1, 0, 0, 1, 0.00064556)
#                 (0, 1, 0, 1, 0, 0.0091675 ) (0, 1, 0, 1, 1, 0.00756625)
#                 (0, 1, 1, 0, 0, 0.00674517) (0, 1, 1, 0, 1, 0.05076165)
#                 (0, 1, 1, 1, 0, 0.00835631) (0, 1, 1, 1, 1, 0.00071063)
#                 (1, 0, 0, 0, 0, 0.01483474) (1, 0, 0, 0, 1, 0.00064556)
#                 (1, 0, 0, 1, 0, 0.0091675 ) (1, 0, 0, 1, 1, 0.00756625)
#                 (1, 0, 1, 0, 0, 0.00674517) (1, 0, 1, 0, 1, 0.05076165)
#                 (1, 0, 1, 1, 0, 0.00835631) (1, 0, 1, 1, 1, 0.00071063)
#                 (1, 1, 0, 0, 0, 0.02985095) (1, 1, 0, 0, 1, 0.49483852)
#                 (1, 1, 0, 1, 0, 0.0046068 ) (1, 1, 0, 1, 1, 0.02405322)
#                 (1, 1, 1, 0, 0, 0.04691337) (1, 1, 1, 0, 1, 0.02574434)
#                 (1, 1, 1, 1, 0, 0.0651882 ) (1, 1, 1, 1, 1, 0.00498148)]

print(f"Best params: {solver_results.params}")
# Best params: {'angles': array([0.27298414, 2.2926187 , 0.        , 0.76391714, 0.15569598,
#                                0.4237506 , 0.93474157, 1.39996954, 1.38701602, 0.36818742]),
#               'hyper_args': array([8.77582845, 7.32430447, 1.02777043])}
```
