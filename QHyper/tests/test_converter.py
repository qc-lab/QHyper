import sympy
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel, BinaryPolynomial, make_quadratic_cqm, BINARY

from QHyper.polynomial import Polynomial
from QHyper.problems.base import Problem
from QHyper.parser import from_sympy
from QHyper.converter import Converter
from QHyper.constraint import Constraint, Operator, MethodsForInequalities


class SimpleProblem(Problem):
    def __init__(self, objective_function,
                 constraints, method_for_inequalities) -> None:
        self.objective_function = objective_function
        self.constraints = constraints
        self.method_for_inequalities = method_for_inequalities

    def get_score(self, result: str, penalty: float = 0) -> float:
        # todo implement
        return 0


def test_example_0():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(variables[0] + variables[1])

    constraint_le = Constraint(objective_function, Polynomial(1),
                               Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "s",)

    constraint_le_1 = Constraint(objective_function, Polynomial(2),
                                 Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "t",)
    problem = SimpleProblem(
        objective_function, [constraint_le, constraint_le_1],
        MethodsForInequalities.SLACKS_LOG_2)

    weights = [2., 4., 8.]
    qubo = Converter.create_qubo(problem, weights)

    expected = Polynomial({
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
    })
    assert qubo==expected


def test_example_1():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(- (2 * variables[0] + 5 * variables[1] + variables[0] * variables[1]))

    constraint_eq_lhs = Polynomial({("x0",): 1, ("x1",): 1})
    constraint_eq = Constraint(constraint_eq_lhs, Polynomial(1), Operator.EQ)

    constraint_le_lhs = Polynomial({("x0",): 5, ("x1",): 2})
    constraint_le = Constraint(constraint_le_lhs, Polynomial(5),
                               Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "s")

    problem = SimpleProblem(
        objective_function, [constraint_eq, constraint_le],
        MethodsForInequalities.SLACKS_LOG_2)
    weights = [1., 2., 3.]
    qubo = Converter.create_qubo(problem, weights)

    expected = Polynomial({
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
    })

    assert qubo == expected


def test_example_2():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(
        5 * variables[0]
        + 2 * variables[1]
        + variables[0] * variables[1]
    )

    constraint_le_lhs = Polynomial({("x0",): 5, ("x1",): 2})
    constraint_le = Constraint(constraint_le_lhs, Polynomial(5),
                               Operator.LE, MethodsForInequalities.UNBALANCED_PENALIZATION)

    weights = [1., 1., 1.]
    problem = SimpleProblem(
        objective_function, [constraint_le],
        MethodsForInequalities.SLACKS_LOG_2)
    qubo = Converter.create_qubo(problem, weights)

    expected = Polynomial({
        ("x0", "x0"): 25,
        ("x0", "x1"): 21,
        ("x0",): -40,
        ("x1", "x1"): 4,
        ("x1",): -16,
        (): 20,
    })

    assert qubo == expected


def test_example_3():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(
        5 * variables[0]
        + 2 * variables[1]
        + variables[0] * variables[1]
    )

    constraint_le_lhs = Polynomial({("x0",): 5, ("x1",): 2})
    constraint_le = Constraint(constraint_le_lhs, Polynomial(5),
                               Operator.LE, MethodsForInequalities.UNBALANCED_PENALIZATION)

    constraint_le_lhs_2 = Polynomial({("x0",): 3, ("x1",): 4})
    constraint_le_2 = Constraint(constraint_le_lhs_2, Polynomial(7),
                                 Operator.LE, MethodsForInequalities.UNBALANCED_PENALIZATION)

    problem = SimpleProblem(
        objective_function, [constraint_le, constraint_le_2],
        MethodsForInequalities.UNBALANCED_PENALIZATION)
    weights = [1., 1., 1., 1., 1.]
    qubo = Converter.create_qubo(problem, weights)

    expected = Polynomial({
        ("x0", "x0"): 34,
        ("x0", "x1"): 45,
        ("x0",): -79,
        ("x1", "x1"): 20,
        ("x1",): -68,
        (): 62,
    })

    assert qubo == expected


def test_example_4():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(
        5 * variables[0]
        + 2 * variables[1]
        + variables[0] * variables[1]
    )

    constraint_eq_lhs = Polynomial({("x0",): 1, ("x1",): 1})
    constraint_eq = Constraint(constraint_eq_lhs, Polynomial(1), Operator.EQ)

    problem = SimpleProblem(
        objective_function, [constraint_eq],
        MethodsForInequalities.UNBALANCED_PENALIZATION)
    weights = [1., 6.]
    qubo = Converter.create_qubo(problem, weights)

    expected = Polynomial({
        ("x0", "x0"): 6,
        ("x0", "x1"): 13,
        ("x0",): -7,
        ("x1", "x1"): 6,
        ("x1",): -10,
        (): 6,
    })

    assert qubo == expected


def test_to_dqm():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(variables[0] + variables[1])

    problem = SimpleProblem(
        objective_function, [],
        MethodsForInequalities.SLACKS_LOG_2)

    dqm = Converter.to_dqm(problem)

    created_dqm = DiscreteQuadraticModel()
    for variable in variables:
        added_variable = created_dqm.add_variable(2, str(variable))
        created_dqm.set_linear(
            added_variable,
            [1.0, 1.0]
        )

    print(created_dqm.variables == dqm.variables)

    assert created_dqm.variables == dqm.variables


def test_to_cqm():
    num_variables = 2
    variables = sympy.symbols(
        " ".join([f"x{i}" for i in range(num_variables)])
    )
    objective_function = from_sympy(variables[0] + variables[1])

    constraint_le = Constraint(objective_function, Polynomial(1),
                               Operator.LE, MethodsForInequalities.SLACKS_LOG_2, "s",)

    problem = SimpleProblem(
        objective_function, [constraint_le],
        MethodsForInequalities.SLACKS_LOG_2)

    cqm = Converter.to_cqm(problem)

    created_cqm = make_quadratic_cqm(
        BinaryPolynomial(
            objective_function.terms,
            BINARY
    ))

    created_variables = objective_function.get_variables()
    created_variables.update(constraint_le.lhs.get_variables())

    for variable in created_variables:
        created_cqm.add_variable(BINARY, str(variable))

    lhs = [tuple([*key, value]) for key, value in constraint_le.lhs.terms.items()]
    created_cqm.add_constraint(lhs, constraint_le.operator.value, label=0)

    assert created_cqm.variables == cqm.variables
