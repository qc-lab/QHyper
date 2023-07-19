from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.problems.network_communities.utils import (
    draw_communities_from_graph,
    write_to_file,
)
from QHyper.solvers.gurobi.gurobi import Gurobi

path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"

# name = "brain"

name = "karate"
folder = "demo/demo_output"
solution_file = f"{folder}/{name}_gurobi_solution.csv"
decoded_solution_file = f"{folder}/{name}_gurobi_decoded_solution.csv"
img_solution_path = f"{folder}/{name}_gurobi.png"


karate_problem = CommunityDetectionProblem(KarateClubNetwork, N_communities=4)
problem = karate_problem
# brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)
# brain_problem = CommunityDetectionProblem(brain_network, N_communities=4)
# problem = brain_problem

gurobi = Gurobi(problem=problem)
solution = gurobi.solve({})
decoded_solution = problem.decode_dummies_solution(solution)

write_to_file(solution, solution_file)
write_to_file(decoded_solution, decoded_solution_file)
draw_communities_from_graph(
    problem=problem, sample=decoded_solution, path=img_solution_path
)