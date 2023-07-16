import ast
import csv
import json
import os
from QHyper.problems.brain_community_detection import (
    BrainCommunityDetectionProblem,
)
import sympy
from QHyper.hyperparameter_gen.parser import Expression


num_cases = 4
path = "QHyper/problems/brain_community_data"
data_name = "Edge_AAL90_Binary"
output_folder = "demo/demo_input"
equation_file = f"{output_folder}/{data_name}_dict.csv"


def safe_open(path, permission):
    """
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


def read_dict():
    eq = []
    with open(equation_file, "r") as f:
        data = f.read()

    eq_parse = json.loads(data)
    equation = {ast.literal_eval(k): v for k, v in eq_parse.items()}
    try:
        equation_expression = Expression(equation)
        return equation_expression
    except Exception as e:
        raise Exception("Error", e)


def dict_to_file(equation_expression: Expression):
    with safe_open(equation_file, "w") as file:
        data = json.dumps({str(k): v for k, v in equation_expression.as_dict().items()})
        file.write(data)


def generate_equation_dict():
    brain = BrainCommunityDetectionProblem(path, data_name, num_cases)

    # Obj. fun. should be empty [] for now until the
    # the issue is resolved
    print(brain.objective_function)

    G = brain.G
    variables = sympy.symbols(" ".join([f"x{i}" for i in range(len(G.nodes))]))

    equation: dict = {}
    for i in G.nodes():
        for j in range(i + 1, len(G.nodes)):
            u_var, v_var = variables[i], variables[j]
            equation[(u_var.name, v_var.name)] = -brain.B[i, j]

    equation_expression = Expression(equation)
    try:
        print(equation_expression)  
        print(equation_expression.as_polynomial())
    finally:
        dict_to_file(equation_expression)

if __name__ == "__main__":
    generate_equation_dict()
    read_dict()
