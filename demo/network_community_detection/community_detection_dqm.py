import networkx as nx
import networkx.algorithms.community as nx_comm
from utils.utils import communities_from_sample, draw_communities

from demo.network_community_detection.util import CWD
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.solvers.dqm.dqm import DQM

folder = f"{CWD}/demo_output"
folder_csv = f"{folder}/csv_files"
name = "karate"
solver = "dqm"

resolution = 0.5

problem = CommunityDetectionProblem(
    KarateClubNetwork(resolution=resolution), communities=2
)

dqm = DQM(problem, time=5)
sampleset = dqm.solve()
sample = sampleset.first.sample

communities = [
    {int(c[1:]) for c in comm}
    for comm in communities_from_sample(sample, problem.cases)
]
modularity = nx_comm.modularity(
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
