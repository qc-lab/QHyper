from typing import cast

import pytest
import sympy

from QHyper.problems.base import Problem
from QHyper.solvers.converter import Converter
from QHyper.util import Constraint, Expression, MethodsForInequalities, Operator


class SimpleProblem(Problem):
    def __init__(self, num_variables) -> None:
        self.num_variables = num_variables
        self.variables = sympy.symbols(
            " ".join([f"x{i}" for i in range(self.num_variables)])
        )
        self.objective_function: Expression = None
        self.constraints: list[Constraint] = []
        self.method_for_inequalities = None

    def set_objective_function(self, expression) -> None:
        self.objective_function = Expression(expression)

    def add_constraint(self, constraint) -> None:
        self.constraints.append(constraint)

    def set_method_for_inequalities(self, method: MethodsForInequalities) -> None:
        self.method_for_inequalities = method

    def get_score(self, result: str, penalty: float = 0) -> float:
        # todo implement
        return 0

def test_example_0():
    problem = SimpleProblem(2)
    expression = problem.variables[0] + problem.variables[1]
    problem.set_objective_function(expression)

    constraint_le_lhs = {("x0",): 1, ("x1",): 1}
    constraint_le = Constraint(constraint_le_lhs, 1, Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "s",)
    problem.add_constraint(constraint_le)
    
    constraint_le_lhs_1 = {("x0",): 1, ("x1",): 1}
    constraint_le_1 = Constraint(constraint_le_lhs_1, 2, Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "t",)
    problem.add_constraint(constraint_le_1)
    
    weights = [2, 4, 8]
    qubo = Converter.create_qubo(problem, weights)
    
    expected = {
        ('s_0', 'x0'): 8,
        ('s_0', 'x1'): 8,
        ('s_0', 's_0'): 4,
        ('s_0', ): -8,
        ('t_0', 'x0'): 16,
        ('t_1', 'x0'): 16,
        ('t_0', 'x1'): 16,
        ('t_1', 'x1'): 16,
        ('t_0', 't_0'): 8,
        ('t_1', 't_1'): 8,
        ('t_0', ): -32,
        ('t_0', 't_1'): 16,
        ('t_1', ): -32,
        ('x0', 'x0'): 12,
        ('x1', 'x1'): 12,
        ('x0', ): -38,
        ('x0', 'x1'): 24,
        ('x1', ): -38,
        (): 36,     
    }
    assert qubo==expected
    

def test_example_1():
    problem = SimpleProblem(2)
    expression = - (2 * problem.variables[0] + 5 * problem.variables[1] + problem.variables[0] * problem.variables[1])
    problem.set_objective_function(expression)

    constraint_eq_lhs = {("x0",): 1, ("x1",): 1}
    constraint_eq = Constraint(constraint_eq_lhs, 1, Operator.EQ)
    problem.add_constraint(constraint_eq)
    
    constraint_le_lhs = {("x0",): 5, ("x1",): 2}
    constraint_le = Constraint(constraint_le_lhs, 5, Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "s")
    problem.add_constraint(constraint_le)

    weights = [1, 2, 3]
    qubo = Converter.create_qubo(problem, weights)

    expected = {
        ("s_0", "x0"): 30,
        ("s_0", "x1"): 12,
        ("s_1", "x0"): 60,
        ("s_2", "x0"): 60,
        ("s_1", "x1"): 24,
        ("s_0", "s_0"): 3,
        ("s_0", "s_1"): 12,
        ("s_0", "s_2"): 12,
        ("s_0",): -30,
        ("s_1", "s_1"): 12,
        ("s_2", "s_2"): 12,
        ("s_1",): -60,
        ("s_1", "s_2"): 24,
        ("s_2",): -60,
        ("s_2", "x1"): 24,
        ("x0", "x0"): 77,
        ("x1", "x1"): 14,
        ("x0",): -156,
        ("x0", "x1"): 63,
        ("x1",): -69,
        (): 77,
    }

    assert qubo == expected


def test_example_2():
    problem = SimpleProblem(2)
    expression = (
        5 * problem.variables[0]
        + 2 * problem.variables[1]
        + problem.variables[0] * problem.variables[1]
    )
    problem.set_objective_function(expression)

    constraint_le_lhs = {("x0",): 5, ("x1",): 2}
    constraint_le = Constraint(constraint_le_lhs, 5, Operator.LE, MethodsForInequalities.UNBALANCED_PENALIZATION)
    problem.add_constraint(constraint_le)

    weights = [1, 1, 1]
    qubo = Converter.create_qubo(problem, weights)

    expected = {
        ("x0", "x0"): 25,
        ("x0", "x1"): 21,
        ("x0",): -40,
        ("x1", "x1"): 4,
        ("x1",): -16,
        (): 20,
    }

    assert qubo == expected


def test_example_3():
    problem = SimpleProblem(2)
    expression = (
        5 * problem.variables[0]
        + 2 * problem.variables[1]
        + problem.variables[0] * problem.variables[1]
    )
    problem.set_objective_function(expression)

    constraint_le_lhs = {("x0",): 5, ("x1",): 2}
    constraint_le = Constraint(constraint_le_lhs, 5, Operator.LE, MethodsForInequalities.UNBALANCED_PENALIZATION)
    problem.add_constraint(constraint_le)

    constraint_le_lhs_2 = {("x0",): 3, ("x1",): 4}
    constraint_le_2 = Constraint(constraint_le_lhs_2, 7, Operator.LE, MethodsForInequalities.UNBALANCED_PENALIZATION)
    problem.add_constraint(constraint_le_2)

    weights = [1, 1, 1, 1, 1]
    qubo = Converter.create_qubo(problem, weights)

    expected = {
        ("x0", "x0"): 34,
        ("x0", "x1"): 45,
        ("x0",): -79,
        ("x1", "x1"): 20,
        ("x1",): -68,
        (): 62,
    }

    assert qubo == expected


def test_example_4():
    problem = SimpleProblem(2)
    expression = (
        5 * problem.variables[0]
        + 2 * problem.variables[1]
        + problem.variables[0] * problem.variables[1]
    )
    problem.set_objective_function(expression)

    constraint_eq_lhs = {("x0",): 1, ("x1",): 1}
    constraint_eq = Constraint(constraint_eq_lhs, 1, Operator.EQ)
    problem.add_constraint(constraint_eq)

    weights = [1, 6]
    qubo = Converter.create_qubo(problem, weights)

    expected = {
        ("x0", "x0"): 6,
        ("x0", "x1"): 13,
        ("x0",): -7,
        ("x1", "x1"): 6,
        ("x1",): -10,
        (): 6,
    }

    assert qubo == expected
