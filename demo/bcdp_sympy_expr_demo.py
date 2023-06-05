import csv
import os
from QHyper.problems.brain_community_detection import (
    BrainCommunityDetectionProblem,
)
import sympy
from sympy.core.expr import Expr
from typing import cast
from QHyper.hyperparameter_gen.parser import Expression
from sympy import sympify


num_cases = 4
path = "QHyper/problems/brain_community_data"
data_name = "Edge_AAL90_Binary"
output_folder = "demo/demo_input"
equation_file = f"{output_folder}/{data_name}_sympy_expression.csv"


def safe_open(path, permission):
    """
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


def read_sympy_equation():
    eq = []
    with open(equation_file, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            eq.append(line)

    equation: Expr = cast(Expr, 0)
    try:
        # Will crash in the sympify function
        # as a function of 4 000 elements will crash while
        # tree parsing in the sympy expression
        equation += sympify(eq[0][0], rational=True)  # will crash here
        equation_expression = Expression(equation)
        return equation_expression
    except Exception as e:
        raise Exception("Error", e)


def sympy_expr_to_file(equation_expression: Expression):
    with safe_open(equation_file, "w") as file:
        file.write(str(equation_expression))


def generate_equation_sympy_expr():
    brain = BrainCommunityDetectionProblem(path, data_name, num_cases)

    # Obj. fun. should be empty [] for now until the
    # the issue is resolved
    print(brain.objective_function)

    G = brain.G
    variables = sympy.symbols(" ".join([f"x{i}" for i in range(len(G.nodes))]))

    equation: Expr = cast(Expr, 0)
    for i in G.nodes():
        for j in range(i + 1, len(G.nodes)):
            u_var, v_var = variables[i], variables[j]
            equation += u_var * v_var * brain.B[i, j]
    equation *= -1

    equation_expression = Expression(equation)
    try:
        # Code will crash here, as printing a sympy expression
        # equation calls the .as_dict() function and for such a big
        # equation (4 000 elements) the sympy's recursion depth is exceeded
        print(equation_expression)  # will crash here
    finally:
        # Will execute successfully as writing a sympy expr. to a file
        # does not require parsing the expr. to a dict form
        sympy_expr_to_file(equation_expression)


if __name__ == "__main__":
    # generate_equation_sympy_expr()

    # The line above commented as the function is already
    # written to a file, so there's no need to generate it
    # (it will save you a couple of min.)
    read_sympy_equation()
