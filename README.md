# <img width="40" alt="qhyper_logo" src="https://user-images.githubusercontent.com/38388283/226841016-711112a8-09d1-4a83-8aab-6e305cb24edb.png" class="center"> QHyper
A software framework for hybrid quantum-classical optimization.

# Introduction

QHyper is a python library which main goal is to make easier conducting experiments.
Focuses on quantum algorithms, allows also to use classical solvers.

# Code example


```
>>> problem = KnapsackProblem(max_weight=2, items=[(1, 2), (1, 2), (1, 1)])

>>> params_config = {
        'angels': [[0.5]*5, [1]*5],
        'hyper_args': [1, 2.5, 2.5],
    }

>>> config = {
        'solver': {
            'type': 'vqa',
            'args': {
                'optimizer': {
                    'type': 'scipy',
                    'maxfun': 200,
                },
                'pqc': {
                    'type': 'hqaoa',
                    'layers': 5,
                },
            }
        },
        'hyper_optimizer': {
            'type': 'random',
            'processes': 5,
            'number_of_samples': 100,
            'bounds': [[1, 10], [1, 10], [1, 10]]
        }
    }

>>> vqa = Solver.from_config(problem, config)

>>> vqa.solve(params_cofing)
{
    'angles': array([0.20700558, 0.85908389, 0.58705246, 0.52192666, 0.7343595 ,
                        0.98882406, 1.00509525, 1.0062573 , 0.9811152 , 1.12500301]),
    'hyper_args': array([1.02005268, 2.10821942, 1.94148846, 2.14637279, 1.88565744])
}
```

# Documentation

Documentation can be found at [qhyper.readthedocs.io](https://qhyper.readthedocs.io/en/latest/)
