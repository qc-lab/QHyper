import csv
import os
from QHyper.problems.brain_community_detection import BrainCommunityDetectionProblem
import sympy
from sympy.core.expr import Expr
from typing import cast
from QHyper.hyperparameter_gen.parser import Expression
from sympy import srepr, sympify


num_cases = 4
path = "QHyper/problems/brain_community_data"
data_name = "Edge_AAL90_Binary"
output_folder = "demo/demo_output"
eq_file = f"{output_folder}/{data_name}_sympy_expression.csv"


def safe_open(path, permission):
    """
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


def read_sympy_equation():
    eq = []

    with open(f"{output_folder}/abc.csv", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            eq.append(line)

    # print(sympify(eq[0][0], rational=True))

    equation: Expr = cast(Expr, 0)
    equation += sympify(eq[0][0], rational=True)
    print(srepr(equation))
    # equation_expression = Expression(equation)

    # return equation_expression


def main():
    brain = BrainCommunityDetectionProblem(path, data_name, num_cases)
    print(brain.objective_function)

    G = brain.G
    variables = sympy.symbols(" ".join([f"x{i}" for i in range(len(G.nodes))]))
    print(variables)

    equation: Expr = cast(Expr, 0)
    for i in G.nodes():
        for j in range(i, len(G.nodes)):
            if i == j:
                continue
            u_var, v_var = variables[i], variables[j]
            print(u_var, v_var, i, j)
            equation += u_var * v_var * brain.B[i, j]
    equation *= -1

    equation_expression = Expression(equation)

    print(equation_expression)  # will crash here
    # with safe_open(f'{output_folder}/{data_name}_sympy_expression.csv', 'w') as file:
    #     file.write(str(equation_expression))


if __name__ == "__main__":
    # main()
    read_sympy_equation()
