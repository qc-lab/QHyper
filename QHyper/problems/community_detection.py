import numpy as np
import sympy
from QHyper.problems.base import Problem
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import cast
import networkx as nx
from dataclasses import dataclass, field

from QHyper.util import VARIABLES
from enum import Enum
from typing import Union


class ObjectiveFunctionFormula(Enum):
    SYMPY_EXPR = 1
    DICT = 2

SYMBOL = sympy.core.symbol.Symbol
VARIABLES_SYMPY = Union[tuple[()], tuple[SYMBOL], tuple[SYMBOL, SYMBOL], tuple[SYMBOL, ...]]


@dataclass
class Network:
    graph: nx.Graph
    modularity_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.modularity_matrix = nx.modularity_matrix(self.graph)


KarateClubNetwork = Network(nx.karate_club_graph())


class BrainNetwork(Network):
    def __init__(
        self, input_data_dir: str, input_data_name: str, delimiter: str = "	"
    ):
        adj_matrix = np.genfromtxt(
            f"{input_data_dir}/{input_data_name}.csv", delimiter=delimiter
        )
        super().__init__(nx.from_numpy_matrix(adj_matrix))


class CommunityDetectionProblem(Problem):
    """
    Problem class instance
    - objective function for network community detection

    Attributes:
    ----------
    N_communities: int
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

    def __init__(self, network_data: Network, N_communities: int = 2, obj_func_formula: ObjectiveFunctionFormula = ObjectiveFunctionFormula.SYMPY_EXPR) -> None:
        """
        Parameters
        ----------
        N_communities: int
            number of communities into which the graph shall be divided
            (default 2)
        """
        self.N_cases = N_communities
        self.G = network_data.graph
        self.B = network_data.modularity_matrix
        self._set_variables()
        self.constraints = []

        if obj_func_formula == ObjectiveFunctionFormula.DICT or isinstance(network_data, BrainNetwork):
            self._set_objective_function_as_dict()
        else:
            self._set_objective_function()

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

    def _set_objective_function_as_dict(self):
        equation: dict [VARIABLES, float] = {}
        for i in self.G.nodes:
            for j in range(i + 1, len(self.G.nodes)):
                u_var, v_var = str(self.variables[i]), str(self.variables[j])
                equation[(u_var, v_var)] = self.B[i, j]
        equation = {key: -1*val for key, val in equation.items()}

        self.objective_function = Expression(equation)
