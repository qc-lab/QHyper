import networkx as nx
from utils.utils import (
    communities_from_sample,
    draw_communities_from_graph,
    write_to_file,
)

from demo.network_community_detection.util import CWD
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.solvers.gurobi.gurobi import Gurobi

folder = f"{CWD}/demo_output"
folder_csv = f"{folder}/csv_files"
name = "karate"
solver = "gurobi"
solution_file = f"{folder_csv}/{name}_{solver}_solution.csv"
decoded_solution_file = f"{folder_csv}/{name}_{solver}_decoded_solution.csv"
img_solution_path = f"{folder}/{name}_{solver}.png"

resolution = 0.5
problem = CommunityDetectionProblem(
    KarateClubNetwork(resolution=resolution), communities=1
)

gurobi = Gurobi(problem=problem)
solution = gurobi.solve({})
decoded_solution = problem.decode_solution(solution)

print(f"solution: {solution}")
print(f"decoded_solution: {decoded_solution}")

modularity = nx.community.modularity(
    problem.G,
    communities=communities_from_sample(decoded_solution, problem.cases),
    resolution=resolution,
)
print(f"{solver} {name} modularity: {modularity}")

write_to_file(solution, solution_file)
write_to_file(decoded_solution, decoded_solution_file)
draw_communities_from_graph(
    problem=problem,
    sample=decoded_solution,
    path=img_solution_path,
    pos=nx.spring_layout(problem.G, seed=123),
    title=f"solver: {solver} mod: {modularity}",
)
