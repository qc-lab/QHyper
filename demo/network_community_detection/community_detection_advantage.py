import networkx as nx
from util import CWD
from utils.utils import (
    communities_from_sample,
    draw_communities
)

from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
    BrainNetwork
)
from QHyper.solvers.advantage import Advantage

folder = f"{CWD}/demo_output"
folder_csv = f"{folder}/csv_files"
name = "karate"
solver = "advantage"



# resolution = 0.93

# problem = CommunityDetectionProblem(
#     network_data=KarateClubNetwork(resolution=resolution), communities=1 
# )

resolution = 3

problem = CommunityDetectionProblem(
    network_data=BrainNetwork(f"{CWD}/brain_networks_data", "example_data", 
                              resolution=resolution), communities=1
)

advantage = Advantage(problem)
sampleset = advantage.solve()
sample = sampleset.first.sample

communities = [
    {int(c[1:]) for c in comm}
    for comm in communities_from_sample(sample, 2)
]
print(communities)
print(problem.G)
modularity = nx.community.modularity(
    problem.G, communities=communities, resolution=resolution
)
print(f"{solver} {name} modularity: {modularity}")

draw_communities(
    problem=problem,
    sample=sample,
    path=f"{folder}/{name}_{solver}.png",
    title=f"solver: {solver} mod: {modularity}",
    pos=nx.spring_layout(problem.G, seed=123),
)

