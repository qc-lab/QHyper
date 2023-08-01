from QHyper.problems.community_detection import (
    BrainNetwork,
    CommunityDetectionProblem,
    KarateClubNetwork,
)

from QHyper.problems.network_communities.utils import (
    draw_communities,
    draw_communities_from_graph,
    write_to_file,
    communities_from_sample
)
from QHyper.solvers.dqm.dqm import DQM
from QHyper.solvers.gurobi.gurobi import Gurobi
import networkx.algorithms.community as nx_comm


folder = "demo/demo_output"
folder_csv = f"{folder}/csv_files"
solver = "gurobi"

path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"
brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)

karate_problem = CommunityDetectionProblem(KarateClubNetwork, N_communities=4)
brain_problem = CommunityDetectionProblem(brain_network, N_communities=4)

problem = karate_problem
name = "karate"

gurobi = Gurobi(problem=problem)
solution = gurobi.solve({})
decoded_solution = problem.decode_dummies_solution(solution)

write_to_file(solution, f"{folder_csv}/{name}_{solver}_solution.csv")
write_to_file(decoded_solution, f"{folder_csv}/{name}_{solver}_decoded_solution.csv")
draw_communities_from_graph(
    problem=problem, sample=decoded_solution, path=f"{folder}/{name}_{solver}.png"
)
modularity = nx_comm.modularity(problem.G, communities=communities_from_sample(decoded_solution, problem.cases))
print(f"{solver} {name} modularity: {modularity}")

problem = brain_problem
name = "brain"

gurobi = Gurobi(problem=problem)
solution = gurobi.solve({"MIPGap": 0.08})
decoded_solution = problem.decode_dummies_solution(solution)

write_to_file(solution, f"{folder_csv}/{name}_{solver}_solution.csv")
write_to_file(decoded_solution, f"{folder_csv}/{name}_{solver}_decoded_solution.csv")
draw_communities_from_graph(
    problem=problem, sample=decoded_solution, path=f"{folder}/{name}_{solver}.png"
)
modularity = nx_comm.modularity(problem.G, communities=communities_from_sample(decoded_solution, problem.cases))
print(f"{solver} {name} modularity: {modularity}")
