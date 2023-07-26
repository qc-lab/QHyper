import gurobipy as gp

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer
from QHyper.solvers.converter import QUBO

from QHyper.problems.community_detection import (
    BrainNetwork,
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.problems.network_communities.utils import (
    draw_communities,
    draw_communities_from_graph,
    write_to_file,
)
from QHyper.solvers.gurobi.gurobi import Gurobi
import os


def calc(vars: dict[str, Any], poly_dict: QUBO) -> Any:
    cost_function: float = 0
    for key, value in poly_dict.items():
        tmp = 1
        for k in key:
            tmp *= vars[str(k)]
        cost_function += tmp * value
    return cost_function


path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"

# name = "brain"

name = "karate"
folder = "demo/demo_output"
# solution_file = f"{folder}/{name}_gurobi_solution.csv"
# decoded_solution_file = f"{folder}/{name}_gurobi_decoded_solution.csv"
img_solution_path = f"{folder}/{name}_gurobi_integer.png"


karate_problem = CommunityDetectionProblem(KarateClubNetwork, N_communities=4)
problem = karate_problem
# brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)
# brain_problem = CommunityDetectionProblem(brain_network, N_communities=4)
# problem = brain_problem

gpm = gp.Model("name")

# vars = {
#     str(var_name): gpm.addVar(lb=0, ub=problem.N_cases-1, vtype=gp.GRB.INTEGER, name=str(var_name))
#     for var_name in problem.discrete_variables
# }
names = [str(var_name) for var_name in problem.discrete_variables]
gpm.addVars(len(problem.discrete_variables), vtype=gp.GRB.INTEGER, lb=[0]*len(problem.discrete_variables), ub=[problem.N_cases-1]*len(problem.discrete_variables), name=names)
gpm.update()

vars = {str(var.VarName): var for var in gpm.getVars()}

objective_function = calc(
    vars, problem.objective_function.as_dict()
)
# print(objective_function)
gpm.setObjective(objective_function, gp.GRB.MINIMIZE)

for i, constraint in enumerate(problem.constraints):
    tmp_constraint = calc(vars, constraint.as_dict())
    print(tmp_constraint)
    gpm.addConstr(tmp_constraint == 0, f"constr_{i}")
    gpm.update()
    print(tmp_constraint)

gpm.optimize()

allvars = gpm.getVars()
solution = {}
for v in allvars:
    solution[v.VarName] = v.X

print(solution)

decoded_solution = problem.decode_dummies_solution(solution)

# write_to_file(solution, solution_file)
# write_to_file(decoded_solution, decoded_solution_file)
# draw_communities_from_graph(
#     problem=problem, sample=decoded_solution, path=img_solution_path
# )
draw_communities(problem=problem, sample=solution, path=img_solution_path)
