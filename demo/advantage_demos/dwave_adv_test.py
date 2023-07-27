# from dwave.system import DWaveSampler
import dimod
import dwave.system
from dimod.generators import and_gate
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
from matplotlib import pyplot as plt
import minorminer
from QHyper.solvers.converter import Converter
import networkx as nx
from dwave.preprocessing import ScaleComposite
import dwave_networkx as dnx

from dwave.system import DWaveSampler

from QHyper.problems.community_detection import CommunityDetectionProblem, KarateClubNetwork
from QHyper.problems.network_communities.utils import (
    draw_communities,
    draw_communities_from_graph,
    write_to_file,
)
from QHyper.problems.community_detection import BrainNetwork
from QHyper.solvers.cqm import CQM
from dimod import BinaryQuadraticModel
from QHyper.solvers.converter import Converter

path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"

# name = "brain"
name = "karate"

folder = "demo/demo_output"
solution_file = f"{folder}/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/{name}_adv_decoded_solution.csv"
img_solution_path = f"{folder}/{name}_adv.png"


# brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)
# brain_problem = CommunityDetectionProblem(brain_network, N_communities=4)
# problem = brain_problem
karate_problem = CommunityDetectionProblem(network_data=KarateClubNetwork, N_communities=2)
problem = karate_problem

adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))
sampler = adv_sampler

# cqm = Converter.to_cqm(problem)
# bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10000)


# embedding = minorminer.find_embedding(
#     dimod.to_networkx_graph(bqm), sampler.to_networkx_graph()
# )

Q = dnx.algorithms.independent_set.maximum_weighted_independent_set_qubo(dimod.to_networkx_graph(bqm))

f = plt.figure()
nx.draw(
    dimod.to_networkx_graph(dimod.BQM.from_qubo(Q)),
    with_labels=True,
    ax=f.add_subplot(111),
)
plt.show()

# embedding = minorminer.find_embedding(
#     Q, sampler.to_networkx_graph()
# )
embedding = minorminer.find_embedding(
    dimod.to_networkx_graph(dimod.BQM.from_qubo(Q)), sampler.to_networkx_graph()
)
print(f"embedding: {embedding}")

sampleset = FixedEmbeddingComposite(
    ScaleComposite(sampler),
    embedding=embedding,
).sample(
    bqm,
    quadratic_range=sampler.properties["extended_j_range"],
    bias_range=sampler.properties["h_range"],
    chain_strength=3,
    num_reads=100,
    auto_scale=False,
    label=f"{name}_problem",
)

sample = sampleset.first.sample
energy = sampleset.first.energy


solution = problem.sort_dummied_encoded_solution(sample)
decoded_solution = problem.decode_dummies_solution(solution)

print(f"solution: {solution}")
print(f"decoded_solution: {decoded_solution}")

write_to_file(solution, solution_file)
write_to_file(decoded_solution, decoded_solution_file)
# draw_communities_from_graph(
#     problem=problem, sample=decoded_solution, path=img_solution_path
# )

COLORS = {
    0: "blue",
    1: "red",
    2: "#2a401f",
    3: "#cce6ff",
    4: "pink",
    5: "#4ebd1a",
    6: "#66ff66",
    7: "yellow",
    8: "#0059b3",
    9: "#703243",
    10: "green",
    11: "black",
    12: "#3495eb",
    13: "#525c4d",
    14: "#1aff1a",
    15: "brown",
    16: "gray",
}

color_map = []
for node in problem.G:
    if node in decoded_solution.keys():
        color_map.append(COLORS[decoded_solution[node]])
    else:
        color_map.append(COLORS[7])

f = plt.figure()
nx.draw(
    problem.G,
    node_color=color_map,
    with_labels=True,
    ax=f.add_subplot(111),
)
f.savefig(img_solution_path)