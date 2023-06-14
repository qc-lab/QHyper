import numpy as np
import sympy
from QHyper.problems.base import Problem
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import List, cast
import networkx as nx
import networkx.algorithms.community as nx_comm
from dataclasses import dataclass, field
from typing import Any

import csv
import time
import os, os.path
from matplotlib import pyplot as plt

from QHyper.problems.community_detection import Network


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


def safe_open(path, permission):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, permission)


@dataclass
class QuantumResultsHandler:
    sample: List[Any] = field(default_factory=list)


    def run_louvain(network_data: Network, seed: int = None) -> tuple(list, float, int, float):
        """
        Run the Louvain community detection algorithm (networkx impl.)
        """
        start = time.time()
        communities = nx_comm.louvain_communities(network_data.graph, seed=seed)  # , seed=123)
        end = time.time()

        run_time = end - start
        modularity_score = nx_comm.modularity(network_data.G, communities)
        N_communities = len(communities)

        return (communities, modularity_score, N_communities, run_time)
    

    


    


