import numpy as np

from QHyper.problems import KnapsackProblem, TSPProblem
from QHyper.problems.knapsack import Item

np.random.seed(1244)


def test_knapsack():
    problem = KnapsackProblem(max_weight=2, items=[(1, 2), (1, 2), (1, 1)])

    assert problem.knapsack.items == [
        Item(weight=1, value=2), Item(weight=1, value=2),
        Item(weight=1, value=1)
    ]

    assert problem.objective_function == {
        ('x0',): -2, ('x1',): -2, ('x2',): -1}

    assert [constraint.lhs for constraint in problem.constraints] == [
        {('x3',): -1, ('x4',): -1, (): 1},
        {('x0',): -1, ('x1',): -1, ('x2',): -1, ('x3',): 1, ('x4',): 2}
    ]


def test_TSP():
    problem = TSPProblem(number_of_cities=4, cities_coords=[(0, 0), (0, 3), (4, 0), (4, 3)])
    assert problem.objective_function == {
        ('x0', 'x15'): 1.0,
        ('x0', 'x7'): 1.0,
        ('x1', 'x14'): 1.0,
        ('x1', 'x6'): 1.0,
        ('x11', 'x4'): 1.0,
        ('x10', 'x5'): 1.0,
        ('x11', 'x12'): 1.0,
        ('x12', 'x3'): 1.0,
        ('x15', 'x8'): 1.0,
        ('x13', 'x2'): 1.0,
        ('x2', 'x5'): 1.0,
        ('x3', 'x4'): 1.0,
        ('x6', 'x9'): 1.0,
        ('x7', 'x8'): 1.0,
        ('x10', 'x13'): 1.0,
        ('x14', 'x9'): 1.0,
        ('x1', 'x15'): 0.8,
        ('x0', 'x14'): 0.8,
        ('x11', 'x13'): 0.8,
        ('x0', 'x6'): 0.8,
        ('x1', 'x7'): 0.8,
        ('x2', 'x4'): 0.8,
        ('x10', 'x12'): 0.8,
        ('x10', 'x4'): 0.8,
        ('x11', 'x5'): 0.8,
        ('x13', 'x3'): 0.8,
        ('x12', 'x2'): 0.8,
        ('x3', 'x5'): 0.8,
        ('x6', 'x8'): 0.8,
        ('x7', 'x9'): 0.8,
        ('x15', 'x9'): 0.8,
        ('x14', 'x8'): 0.8,
        ('x0', 'x13'): 0.6,
        ('x0', 'x5'): 0.6,
        ('x1', 'x12'): 0.6,
        ('x1', 'x4'): 0.6,
        ('x10', 'x15'): 0.6,
        ('x10', 'x7'): 0.6,
        ('x11', 'x14'): 0.6,
        ('x12', 'x9'): 0.6,
        ('x11', 'x6'): 0.6,
        ('x13', 'x8'): 0.6,
        ('x14', 'x3'): 0.6,
        ('x15', 'x2'): 0.6,
        ('x2', 'x7'): 0.6,
        ('x3', 'x6'): 0.6,
        ('x4', 'x9'): 0.6,
        ('x5', 'x8'): 0.6,
    }

    assert [constraint.lhs for constraint in problem.constraints] == [
        {('x0',): -1, ('x4',): -1, ('x8',): -1, ('x12',): -1, (): 1},
        {('x1',): -1, ('x5',): -1, ('x9',): -1, ('x13',): -1, (): 1},
        {('x2',): -1, ('x6',): -1, ('x10',): -1, ('x14',): -1, (): 1},
        {('x3',): -1, ('x7',): -1, ('x11',): -1, ('x15',): -1, (): 1},
        {('x0',): -1, ('x1',): -1, ('x2',): -1, ('x3',): -1, (): 1},
        {('x4',): -1, ('x5',): -1, ('x6',): -1, ('x7',): -1, (): 1},
        {('x8',): -1, ('x9',): -1, ('x10',): -1, ('x11',): -1, (): 1},
        {('x12',): -1, ('x13',): -1, ('x14',): -1, ('x15',): -1, (): 1}
    ]
