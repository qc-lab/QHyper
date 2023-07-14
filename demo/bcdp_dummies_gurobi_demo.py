import os
from matplotlib import pyplot as plt
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    ObjFunFormula as off,
)
from typing import Any
import gurobipy as gp
from QHyper.problems.network_communities.utils import COLORS
import networkx as nx

from QHyper.util import QUBO

from QHyper.problems.community_detection import BrainNetwork


path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"

name = "brain"


def safe_open(path: str, permission: str) -> Any:
    """
    Open "path" for writing, creating any parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


def write_to_file(solution: Any, solution_file: str) -> None:
    with safe_open(solution_file, "w") as file:
        file.write(str(solution))


def calc(vars: dict[str, Any], poly_dict: QUBO) -> Any:
    cost_function: float = 0
    for key, value in poly_dict.items():
        tmp = 1
        for k in key:
            tmp *= vars[k]
        cost_function += tmp * value
    return cost_function


brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)
brain_problem = CommunityDetectionProblem(
    brain_network, N_communities=2, obj_func_formula=off.DICT
)
problem = brain_problem

gpm = gp.Model("BrainProblem")

# Variables
vars = {
    str(var_name): gpm.addVar(vtype=gp.GRB.BINARY, name=str(var_name))
    for _, v in problem.dummy_coefficients.items()
    for var_name in v
}

# Objective function
objective_function = calc(vars, problem.objective_function.as_dict())
gpm.setObjective(objective_function, gp.GRB.MINIMIZE)

# ONE-HOT encoding constraints
for i, constraint in enumerate(problem.constraints):
    tmp_constraint = calc(vars, constraint.as_dict())
    gpm.addConstr(tmp_constraint == 0, f"constr_{i}")
    gpm.update()
    print(tmp_constraint)

gpm.optimize()

allvars = gpm.getVars()
solution = {}
for v in allvars:
    solution[v.VarName] = v.X

print("-----Encoded solution-----")
keyorder = [v for k, dummies in problem.dummy_coefficients.items() for v in dummies]
d = solution
solution = {str(k): d[str(k)] for k in keyorder if str(k) in d}

solution_file = f"demo/demo_output/{name}_gurobi_solution.csv"
write_to_file(solution, solution_file)

# DECODING
decoded_solution = problem.decode_dummies_solution(solution)

color_map = []
for node in problem.G:
    color_map.append(COLORS[decoded_solution[node]])

print("-------Decoded solution------")
keyorder = [int(str(v)[len('x'):]) for v in problem.variables]
d = decoded_solution
decoded_solution = {k: d[k] for k in keyorder if k in d}

decoded_solution_file = f"demo/demo_output/{name}_gurobi_decoded_solution.csv"
write_to_file(decoded_solution, decoded_solution_file)

folder = "demo/demo_output"
data_name = f"{name}_gurobi"
path = f"{folder}/{data_name}.png"

f = plt.figure()
nx.draw(
    problem.G,
    node_color=color_map,
    with_labels=True,
    ax=f.add_subplot(111),
)
f.savefig(path)
