from QHyper.problems.community_detection import BrainNetwork, CommunityDetectionProblem
from QHyper.problems.network_communities.utils import draw_communities
from QHyper.solvers.dqm.dqm import DQM

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
solver = "dqm"

path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"
brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)

karate_problem = CommunityDetectionProblem(KarateClubNetwork, N_communities=4)
brain_problem = CommunityDetectionProblem(brain_network, N_communities=4)


problem = karate_problem
name = "karate"

dqm = DQM(problem, time=5)
sampleset = dqm.solve()
sample = sampleset.first.sample

draw_communities(
    problem=problem, sample=sample, path=f"{folder}/{name}_{solver}.png"
)
communities = [{int(c[len('x'):]) for c in comm} for comm in communities_from_sample(sample, problem.N_cases)]
modularity = nx_comm.modularity(problem.G, communities=communities)
print(f"{solver} {name} modularity: {modularity}")

problem = brain_problem
name = "brain"

dqm = DQM(problem, time=5)
sampleset = dqm.solve()
sample = sampleset.first.sample

draw_communities(
    problem=problem, sample=sample, path=f"{folder}/{name}_{solver}.png"
)
communities = [{int(c[len('x'):]) for c in comm} for comm in communities_from_sample(sample, problem.N_cases)]
modularity = nx_comm.modularity(problem.G, communities=communities)
print(f"{solver} {name} modularity: {modularity}")
