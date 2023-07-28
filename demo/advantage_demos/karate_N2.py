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
img_solution_path = f"{folder}/{name}_adv.png"


def create_qubo(problem: Problem) -> QUBO:
    results: dict[VARIABLES, float] = {}
    for key, value in problem.objective_function.as_dict().items():
        if key in results:
            results[key] += value
        else:
            results[key] = value
    return results


karate_problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork, N_communities=2
)
problem = karate_problem

adv_sampler = DWaveSampler(solver=dict(topology__type="pegasus"))
sampler = adv_sampler
solver = "adv"

# qubo = Converter.create_qubo(problem, [1]+[0]*34)
# qubo = create_qubo(problem)
# binary_polynomial = dimod.BinaryPolynomial(
#     problem.objective_function.as_dict(), dimod.BINARY
# )
# bqm = dimod.make_quadratic(binary_polynomial, 5.0, dimod.BINARY)
# qubo = Converter.to_qubo(problem)
binary_polynomial = dimod.BinaryPolynomial(
    problem.objective_function.as_dict(), dimod.BINARY
)
cqm = dimod.make_quadratic_cqm(binary_polynomial)

# todo this cqm can probably be initialized in some other way
# for var in problem.discrete_variables:
#     if str(var) not in cqm.variables:
#         cqm.add_variable(dimod.BINARY, str(var))

bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)

# bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
# bqm.add_variables_from('BINARY', [problem.discrete_variables])
# print(f"bqm: {bqm}\n\n")
# print(bqm.variables)

# bqm = dimod.BQM.from_qubo(problem.objective_function.as_dict())
# print(bqm.quadratic)
# print(bqm.variables)
# print(bqm.adj)
# print(bqm.num_variables)
# print(bmq.num_inter)

f = plt.figure()
nx.draw(
    dimod.to_networkx_graph(bqm),
    with_labels=True,
    ax=f.add_subplot(111),
)
plt.show()

# embedding = minorminer.find_embedding(
#     dimod.to_networkx_graph(bqm), sampler.to_networkx_graph()
# )
sampleset = EmbeddingComposite(adv_sampler).sample(bqm)

sample = sampleset.first.sample
energy = sampleset.first.energy

# solution = problem.sort_dummied_encoded_solution(sample)
# decoded_solution = problem.decode_dummies_solution(solution)
solution = sample
decoded_solution = {int(str(key)[len('x'):]): val for key, val in solution.items()}


print(f"solution: {solution}\n\n")
print(f"decoded_solution: {decoded_solution}\n\n")

write_to_file(solution, solution_file)
write_to_file(decoded_solution, decoded_solution_file)
# draw_communities_from_graph(
#     problem=problem, sample=decoded_solution, path=img_solution_path
# )
try:
    modularity = nx_comm.modularity(
        problem.G,
        communities=communities_from_sample(decoded_solution, problem.N_cases),
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
f.savefig(img_solution_path)

communities_class = nx_comm.louvain_communities(problem.G, resolution=0.5)#, seed=123)
print(communities_class)
louvain_mod = nx_comm.modularity(problem.G, communities_class)
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
