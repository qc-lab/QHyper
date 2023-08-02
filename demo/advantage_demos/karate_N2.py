# from dwave.system import DWaveSampler
import dimod
import dwave.system
from dimod.generators import and_gate
from dwave.system import (
    DWaveSampler,
    EmbeddingComposite,
    FixedEmbeddingComposite,
)
from matplotlib import pyplot as plt
import minorminer
from QHyper.problems.base import Problem
from QHyper.solvers.converter import Converter
import networkx as nx
from dwave.preprocessing import ScaleComposite
import dwave_networkx as dnx
import networkx.algorithms.community as nx_comm

from dwave.system import DWaveSampler

from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)
from QHyper.problems.network_communities.utils import (
    communities_from_sample,
    draw_communities,
    draw_communities_from_graph,
    write_to_file,
)
from QHyper.problems.community_detection import BrainNetwork
from QHyper.solvers.cqm import CQM
from dimod import BinaryQuadraticModel
from QHyper.solvers.converter import Converter
from QHyper.problems.network_communities.utils import COLORS
from QHyper.util import QUBO, VARIABLES

path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"

name = "karate"

folder = "demo/demo_output"
solution_file = f"{folder}/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/{name}_adv_decoded_solution.csv"
# img_solution_path = f"{folder}/{name}_adv.png"


karate_problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork, N_communities=1
)
problem = karate_problem

adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))
sampler = adv_sampler
solver = "adv"

binary_polynomial = dimod.BinaryPolynomial(
    problem.objective_function.as_dict(), dimod.BINARY
)
cqm = dimod.make_quadratic_cqm(binary_polynomial)
bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)


k = 4
for i in range(k):
    sampleset = EmbeddingComposite(adv_sampler).sample(bqm)
    sample = sampleset.first.sample
    nergy = sampleset.first.energy

    solution = problem.sort_dummied_encoded_solution(sample)
    decoded_solution = {int(str(key)[len('x'):]): val for key, val in solution.items()}
    write_to_file(solution, solution_file)
    write_to_file(decoded_solution, decoded_solution_file)

    try:
        modularity = nx_comm.modularity(
            problem.G,
            communities=communities_from_sample(decoded_solution, problem.cases+1),
            resolution=0.5
        )
        print(f"modularity: {modularity}")
    except Exception as e:
        print(f"exception: {e}")
        modularity = "exception"

    color_map = []
    for node in problem.G:
        if node in decoded_solution.keys():
            color_map.append(COLORS[decoded_solution[node]])
        else:
            color_map.append(COLORS[7])

    f = plt.figure()
    nx.draw(
        problem.G,
        pos=nx.spring_layout(problem.G, seed=123),
        node_color=color_map,
        with_labels=True,
        ax=f.add_subplot(111),
    )
    plt.title(f"solver: {solver} mod: {modularity}")
    f.savefig(f"{folder}/{name}_adv_{i}.png")


communities_class = nx_comm.louvain_communities(problem.G, resolution=0.5)#, seed=123)
print(communities_class)
louvain_mod = nx_comm.modularity(problem.G, communities_class, resolution=0.5)
print(f"modularity_lovain: {louvain_mod}")

color_map = []
for node in problem.G:
    if node in communities_class[0]:
        color_map.append("red")
    elif node in communities_class[1]:
        color_map.append("blue")
    else:
        color_map.append(COLORS[7])

f = plt.figure()
nx.draw(
    problem.G,
    pos=nx.spring_layout(problem.G, seed=123),
    node_color=color_map,
    with_labels=True,
    ax=f.add_subplot(111),
)
plt.title(f"solver: louvain_method mod: {louvain_mod}")
f.savefig(f"{folder}/{name}_louvain.png")
