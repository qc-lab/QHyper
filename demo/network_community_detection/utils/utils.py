import csv
import os
import os.path
from dataclasses import dataclass, field
from typing import Any, List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from QHyper.problems.community_detection import CommunityDetectionProblem

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


def safe_open(path: str, permission: str) -> Any:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


def write_to_file(solution: Any, solution_file_path: str) -> None:
    with safe_open(solution_file_path, "w") as file:
        file.write(str(solution))


@dataclass
class ResultsFrame:
    samples: List[Any] = field(default_factory=list)
    run_times: List[float] = field(default_factory=list)

    communities: List[Any] = field(default_factory=list, init=False)
    communities_counts: List[Any] = field(default_factory=list, init=False)
    N_communities: List[int | float] = field(default_factory=list, init=False)
    modularity_scores: List[float] = field(default_factory=list, init=False)

    # QA-specific attribute
    energies: List[float] = field(default_factory=list, init=False)

    def calculate_append_communities(
        self, sample: dict, N_communities: int
    ) -> None:
        communities = []
        for k in range(N_communities):
            comm = []
            for i in sample:
                if sample[i] == k:
                    comm.append(i)
            communities.append(set(comm))

        self.communities.append(communities)
        self.N_communities.append(len(communities))

    def calculate_append_communities_counts(
        self, sample: dict, N_communities: int
    ) -> None:
        counts = np.zeros(N_communities)
        for i in sample:
            counts[sample[i]] += 1

        # Assert each node is assigned to a community
        assert np.sum(counts) == len(sample)
        self.communities_counts.append(counts)

    def louvain_communities_to_sample_like(self, louvain_communities) -> dict:
        sample_like = {
            node_i: comm_i
            for comm_i, comms_set in enumerate(louvain_communities)
            for node_i in comms_set
        }
        return dict(sorted(sample_like.items()))


def communities_from_sample(sample: dict, N_communities: int) -> list:
    communities: list = []
    for k in range(N_communities):
        comm = []
        for i in sample:
            if sample[i] == k:
                comm.append(i)
        communities.append(set(comm))

    return communities


def draw_communities(
    problem: CommunityDetectionProblem, sample: dict, path: str, **kwargs
) -> None:
    color_map = []
    char = list(sample.keys())[0][0]
    for node in problem.G:
        color_map.append(COLORS[sample[char + str(node)]])

    pos = kwargs.get("pos") if "pos" in kwargs else None
    f = plt.figure()
    nx.draw(
        problem.G,
        pos=pos,
        node_color=color_map,
        with_labels=True,
        ax=f.add_subplot(111),
    )
    if "title" in kwargs:
        plt.title(kwargs.get("title"))
    try:
        f.savefig(path)
    except Exception:
        plt.show()


def draw_communities_from_graph(
    problem: CommunityDetectionProblem, sample: dict, path: str, **kwargs
) -> None:
    color_map = []
    for node in problem.G:
        color_map.append(COLORS[sample[node]])

    pos = kwargs.get("pos") if "pos" in kwargs else None
    f = plt.figure()
    nx.draw(
        problem.G,
        pos=pos,
        node_color=color_map,
        with_labels=True,
        ax=f.add_subplot(111),
    )
    if "title" in kwargs:
        plt.title(kwargs.get("title"))
    try:
        f.savefig(path)
    except Exception:
        plt.show()


def communities_to_csv(
    problem: CommunityDetectionProblem,
    sample: dict,
    path: str,
    delimiter: str = ",",
) -> None:
    clus = np.zeros((len(problem.G.nodes), 2))

    for i, node in enumerate(problem.G):
        clus[i, 0] = node
        clus[i, 1] = sample["x" + str(node)]

    np.savetxt(path, clus, delimiter=delimiter)


def results_to_csv(
    qa_results: ResultsFrame, cl_results: ResultsFrame, path: str
) -> None:
    assert len(qa_results.communities) == len(cl_results.communities)

    headers = [
        "modularity_classical",
        "modularity_quantum",
        "communities",
        "run_time_dwave",
        "energy",
        "counts",
        "sample",
        "communities_class_louvain",
        "time_louvain",
    ]
    with safe_open(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for i in range(len(qa_results.communities)):
            data = [
                qa_results.modularity_scores[i],
                cl_results.modularity_scores[i],
                qa_results.communities[i],
                qa_results.run_times[i],
                qa_results.energies[i],
                qa_results.communities_counts[i],
                qa_results.samples[i],
                cl_results.communities[i],
                cl_results.run_times[i],
            ]
            writer.writerow(data)
    file.close()
