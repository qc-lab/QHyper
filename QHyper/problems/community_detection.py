import numpy as np
import sympy
from QHyper.problems.base import Problem
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import cast
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class Network:
    graph: nx.Graph
    modularity_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.modularity_matrix = nx.modularity_matrix(self.G)

KarateClub = Network(nx.karate_club_graph())

@dataclass
class BrainNetwork(Network):
    input_data_dir: str
    input_data_name: str
    delimiter: str = "	"
    adjacency_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.adjacency_matrix = np.genfromtxt(f"{self.input_data_dir}/{self.input_data_name}.csv", delimiter=self.delimiter)
        super().__init__(nx.from_numpy_matrix(self.adjacency_matrix))


class CommunityDetectionProblem(Problem):
    """
    Problem class instance
    - objective function for network community detection

    Attributes:
    ----------
    num_communities: int
        number of communities into which the graph shall be divided
    G: networkx graph
        networkx implementation of graph
    B: networkx modularity matrix
        networkx implementation of modularity matrix

    objective_function : Expression
        objective function in SymPy syntax
    constraints : list[Expression]
        list of problem constraints in SymPy syntax, (default [])
    variables : int
        number of qubits in the circuit, equal to number of nodes
        in the graph
    """

    def __init__(self, num_communities: int = 2) -> None:
        """
        Parameters
        ----------
        num_communities: int
            number of communities into which the graph shall be divided
            (default 2)
        """
        self.num_communities = num_communities
        self.G = nx.karate_club_graph()
        self.B = nx.modularity_matrix(self.G)
        self._set_variables()
        self._set_objective_function()
        self.constraints = []

    def _set_variables(self) -> None:
        """
        Set the variables in SymPy syntax
        """
        self.variables = sympy.symbols(
            " ".join([f"x{i}" for i in range(len(self.G.nodes))])
        )

    def _set_objective_function(self) -> None:
        """
        Create the objective function defined in SymPy syntax
        """
        # xs = [f"x{i}" for i in range(len(self.G.nodes))]
        equation: Expr = cast(Expr, 0)
        for i in self.G.nodes():
            for j in range(i + 1, len(self.G.nodes)):
                u_var, v_var = self.variables[i], self.variables[j]
                equation += u_var * v_var * self.B[i, j]
        equation *= -1

        self.objective_function = Expression(equation)



    
