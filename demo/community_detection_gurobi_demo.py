import os
from matplotlib import pyplot as plt
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
    ObjFunFormula as off,
)
from typing import Any
import gurobipy as gp
from QHyper.problems.network_communities.utils import COLORS
import networkx as nx

from QHyper.util import QUBO


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


karate_problem = CommunityDetectionProblem(
    KarateClubNetwork, N_communities=2, obj_func_formula=off.DICT
)
problem = karate_problem

gpm = gp.Model("KarateProblem")

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


solution_file = "demo/demo_output/brain_gurobi_solution.csv"
write_to_file(solution, solution_file)


# DECODING
decoded_solution = karate_problem.decode_dummies_solution(solution)

color_map = []
for node in problem.G:
    color_map.append(COLORS[decoded_solution[node]])


decoded_solution_file = "demo/demo_output/brain_gurobi_decoded_solution.csv"
write_to_file(decoded_solution, decoded_solution_file)

# with safe_open(solution_file, "r") as f:
#     solution = f.read()

# solution = eval(solution)

folder = "demo/demo_output"
data_name = "bcdp_gurobi_binary"
path = f"{folder}/{data_name}.png"

f = plt.figure()
nx.draw(
    problem.G,
    node_color=color_map,
    with_labels=True,
    ax=f.add_subplot(111),
)
f.savefig(path)
