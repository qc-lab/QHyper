import numpy as np

from QHyper.problems import KnapsackProblem, TSPProblem

np.random.seed(1244)


def test_knapsack():
    from QHyper.problems.knapsack import Item

    problem = KnapsackProblem(max_weight=2, items=[(1, 2), (1, 2), (1, 1)])

    assert problem.knapsack.items == [
        Item(weight=1, value=2), Item(weight=1, value=2),
        Item(weight=1, value=1)
    ]

    assert problem.objective_function.dictionary == {
        ('x0',): -2, ('x1',): -2, ('x2',): -1}

    assert [constraint.lhs for constraint in problem.constraints] == [
        {('x3',): -1, ('x4',): -1, (): 1},
        {('x0',): -1, ('x1',): -1, ('x2',): -1, ('x3',): 1, ('x4',): 2}
    ]


def test_TSP():
    problem = TSPProblem(number_of_cities=3)

    assert problem.objective_function.dictionary == {
        ('x0', 'x4'): 0.591474650245022,
        ('x0', 'x5'): 1.0,
        ('x0', 'x7'): 0.591474650245022,
        ('x0', 'x8'): 1.0,
        ('x1', 'x3'): 0.591474650245022,
        ('x1', 'x5'): 0.842995170488782,
        ('x1', 'x6'): 0.591474650245022,
        ('x1', 'x8'): 0.842995170488782,
        ('x2', 'x3'): 1.0,
        ('x2', 'x4'): 0.842995170488782,
        ('x2', 'x6'): 1.0,
        ('x2', 'x7'): 0.842995170488782,
        ('x3', 'x7'): 0.591474650245022,
        ('x3', 'x8'): 1.0,
        ('x4', 'x6'): 0.591474650245022,
        ('x4', 'x8'): 0.842995170488782,
        ('x5', 'x6'): 1.0,
        ('x5', 'x7'): 0.842995170488782
    }

    assert [constraint.lhs for constraint in problem.constraints] == [
        {('x0',): -1, ('x3',): -1, ('x6',): -1, (): 1},
        {('x1',): -1, ('x4',): -1, ('x7',): -1, (): 1},
        {('x2',): -1, ('x5',): -1, ('x8',): -1, (): 1},
        {('x6',): -1, ('x7',): -1, ('x8',): -1, (): 1},
    ]
