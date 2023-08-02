import dimod
import networkx as nx
import networkx.algorithms.community as nx_comm
from dwave.system import DWaveSampler, EmbeddingComposite
from matplotlib import pyplot as plt

from demo.network_community_detection.utils.utils import (
    COLORS,
    communities_from_sample,
    write_to_file,
)
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
    KarateClubNetwork,
)

name = "karate"
folder = "demo/network_community_detectio/demo_output"
solution_file = f"{folder}/csv_files/{name}_adv_solution.csv"
decoded_solution_file = f"{folder}/csv_files/{name}_adv_decoded_solution.csv"
# img_solution_path = f"{folder}/{name}_adv.png"

resolution = 0.5

problem = CommunityDetectionProblem(
    network_data=KarateClubNetwork(resolution=1), N_communities=1
)

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

    print(f"sample: {sample}")

    solution = problem.sort_dummied_encoded_solution(sample)
    decoded_solution = {
        int(str(key)[len("x") :]): val for key, val in solution.items()
    }
    write_to_file(solution, solution_file)
    write_to_file(decoded_solution, decoded_solution_file)
    print(f"solution: {solution}\n")
    print(f"decoded_solution: {decoded_solution}\n\n")

    try:
        modularity = nx_comm.modularity(
            problem.G,
            communities=communities_from_sample(
                decoded_solution, problem.cases + 1
            ),
            resolution=resolution,
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


communities_class = nx_comm.louvain_communities(
    problem.G, resolution=resolution
)  # , seed=123)
louvain_mod = nx_comm.modularity(
    problem.G, communities_class, resolution=resolution
)
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
