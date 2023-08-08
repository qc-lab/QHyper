import networkx as nx
from util import CWD
from utils.utils import (
    communities_from_sample,
    draw_communities_from_graph,
    write_to_file,
)

from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.solvers.cqm import CQM

name = "karate"
solver = "cqm"

folder = f"{CWD}/demo_output"
solution_file = f"{folder}/csv_files/{name}_cqm_solution.csv"
decoded_solution_file = f"{folder}/csv_files/{name}_cqm_decoded_solution.csv"
img_solution_path = f"{folder}/{name}_{solver}.png"

resolution = 0.5

problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=resolution), communities=2
)

cqm = CQM(problem, time=5)
solution = cqm.solve()
solution = problem.sort_encoded_solution(solution)
decoded_solution = problem.decode_solution(solution)

print(f"solution: {solution}")
print(f"decoded_solution: {decoded_solution}")

communities = [
    {int(c) for c in comm}
    for comm in communities_from_sample(decoded_solution, problem.cases)
]
modularity = nx.community.modularity(
    problem.G, communities=communities, resolution=resolution
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
