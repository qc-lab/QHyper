import numpy as np
import networkx as nx

from QHyper.problems.tsp import TravelingSalesmanProblem
from QHyper.problems.community_detection import (
    CommunityDetectionProblem, Network)
from QHyper.problems.knapsack import Item, KnapsackProblem

np.random.seed(1244)


def test_knapsack():
    problem = KnapsackProblem(max_weight=2, item_weights=[1, 1, 1],
                              item_values=[2, 2, 1])

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
    problem = TravelingSalesmanProblem(
        number_of_cities=4, cities_coords=[(0, 0), (0, 3), (4, 0), (4, 3)])
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


def test_CDP():
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
    problem = CommunityDetectionProblem(Network(G), 2)

    assert problem.objective_function == {
        ('s0', 's2'): -1.0, ('s1', 's3'): -1.0,
        ('s0', 's4'): 1.0, ('s1', 's5'): 1.0,
        ('s0', 's6'): -1.0, ('s1', 's7'): -1.0,
        ('s2', 's4'): -1.0, ('s3', 's5'): -1.0,
        ('s2', 's6'): 1.0, ('s3', 's7'): 1.0,
        ('s4', 's6'): -1.0, ('s5', 's7'): -1.0
    }

    assert [constraint.lhs for constraint in problem.constraints] == [
        {('s0',): 1.0, ('s1',): 1.0, (): -1.0},
        {('s2',): 1.0, ('s3',): 1.0, (): -1.0},
        {('s4',): 1.0, ('s5',): 1.0, (): -1.0},
        {('s6',): 1.0, ('s7',): 1.0, (): -1.0}
     ]

    problem_no_one_hot = CommunityDetectionProblem(Network(G), 2, False)

    assert problem_no_one_hot.objective_function == {
        ('x0', 'x0'): 0.5, ('x0', 'x1'): -1.0,
        ('x0', 'x2'): 1.0, ('x0', 'x3'): -1.0,
        ('x1', 'x1'): 0.5, ('x1', 'x2'): -1.0,
        ('x1', 'x3'): 1.0, ('x2', 'x2'): 0.5,
        ('x2', 'x3'): -1.0, ('x3', 'x3'): 0.5
    }

    assert problem_no_one_hot.constraints == []
