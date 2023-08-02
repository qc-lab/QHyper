import numpy as np
import sympy
from QHyper.problems.base import Problem
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import cast
import networkx as nx
from dataclasses import dataclass, field
from typing import Iterable, Tuple

from QHyper.util import VARIABLES


@dataclass
class Network:
    graph: nx.Graph
    resolution: float = field(default=1)
    modularity_matrix: np.ndarray = field(init=False)
    weight: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.modularity_matrix = self.calculate_modularity_matrix()

    def calculate_modularity_matrix(self) -> np.ndarray:
        adj_matrix: np.ndarray = nx.to_numpy_array(
            self.graph, weight=self.weight
        )
        degree_matrix: np.ndarray = adj_matrix.sum(axis=1)
        m: int = nx.number_of_edges(self.graph)
        return adj_matrix - self.resolution * np.outer(
            degree_matrix, degree_matrix
        ) / (2 * m)


KarateClubNetwork = Network(nx.karate_club_graph())


class BrainNetwork(Network):
    def __init__(
        self,
        input_data_dir: str,
        input_data_name: str,
        delimiter: str = "	",
        resolution: int = 1,
    ):
        adj_matrix = np.genfromtxt(
            f"{input_data_dir}/{input_data_name}.csv", delimiter=delimiter
        )
        super().__init__(
            nx.from_numpy_matrix(adj_matrix), resolution=resolution
        )


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

    def __init__(self, network_data: Network, N_communities: int = 2) -> None:
        """
        Parameters
        ----------
        N_communities: int
            number of communities into which the graph shall be divided
            (default 2)
        """
        if N_communities < 1:
            raise Exception(
                "Number of communities must be greater than or equal to 1"
            )
        self.cases: int = N_communities
        self.G: nx.Graph = network_data.graph
        self.B: np.ndarray = network_data.modularity_matrix
        self.variables: tuple[
            sympy.Symbol
        ] = self._encode_discretes_to_one_hots()
        self._set_objective_function()
        self._set_one_hot_constraints()

    def encode_discrete_to_one_hot(
        self, discrete_variable: sympy.Symbol, case_value: int
    ) -> sympy.Symbol:
        discrete_id = int(str(discrete_variable)[len("x") :])
        id = discrete_id * self.cases + case_value
        return sympy.symbols(f"s{id}")

    def decode_one_hot_to_discrete(
        self, variable: sympy.Symbol
    ) -> sympy.Symbol:
        id = int(str(variable)[len("s") :])
        discrete_id = int(id // self.cases)
        return sympy.symbols(f"x{discrete_id}")

    def decode_case_value_from_one_hot(self, variable: sympy.Symbol) -> int:
        id = int(str(variable)[len("s") :])
        case_value = id % self.cases
        return case_value

    def _encode_discretes_to_one_hots(self) -> tuple[sympy.Symbol]:
        one_hots: tuple[sympy.Symbol] = sympy.symbols(
            " ".join(
                [
                    str(self.encode_discrete_to_one_hot(var, case_val))
                    for var in self._get_discrete_variable_representation()
                    for case_val in range(self.cases)
                ]
            )
        )
        return one_hots

    def yield_variables_cases(self) -> Iterable[Tuple[sympy.Symbol, ...]]:
        """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ..."""
        return zip(*[iter(self.variables)] * self.cases)

    def _get_discrete_variable_representation(self) -> tuple[sympy.Symbol]:
        return sympy.symbols(
            " ".join([f"x{i}" for i in range(len(self.G.nodes))])
        )

    def _set_objective_function(self) -> None:
        equation: dict[VARIABLES, float] = {}
        for i in self.G.nodes:
            for j in range(i + 1, len(self.G.nodes)):
                for case_val in range(self.cases):
                    x_i, x_j = sympy.symbols(f"x{i}"), sympy.symbols(f"x{j}")
                    s_i = str(self.encode_discrete_to_one_hot(x_i, case_val))
                    s_j = str(self.encode_discrete_to_one_hot(x_j, case_val))
                    equation[(s_i, s_j)] = self.B[i, j]

        equation = {key: -1 * val for key, val in equation.items()}

        self.objective_function = Expression(equation)

    def _set_one_hot_constraints(self) -> None:
        self.constraints: list[Expression] = []
        ONE_HOT_CONST = -1

        dummies: Iterable[Tuple[sympy.Symbol, ...]]
        for dummies in self.yield_variables_cases():
            expression: Expr = cast(Expr, 0)
            dummy: sympy.Symbol
            for dummy in dummies:
                expression += dummy
            expression += ONE_HOT_CONST
            self.constraints.append(Expression(expression))

    def decode_dummies_solution(self, solution: dict) -> dict:
        ONE_HOT_VALUE = 1.0
        decoded_solution: dict = {}

        for variable, value in solution.items():
            _, id = variable[0], int(variable[len("s") :])
            if value == ONE_HOT_VALUE:
                case_value = id % self.cases
                variable_id = id // self.cases
                decoded_solution[variable_id] = case_value

        decoded_solution = self.sort_decoded_solution(decoded_solution)
        return decoded_solution

    def sort_dummied_encoded_solution(self, dummies_solution: dict) -> dict:
        keyorder = self.variables
        solution_by_keyorder: dict = {
            str(k): dummies_solution[str(k)]
            for k in keyorder
            if str(k) in dummies_solution
        }
        return solution_by_keyorder

    def sort_decoded_solution(self, decoded_solution: dict) -> dict:
        keyorder = [int(str(v)[len("x") :]) for v in self.variables]
        decoded_solution_by_keyorder: dict = {
            k: decoded_solution[k] for k in keyorder if k in decoded_solution
        }
        return decoded_solution_by_keyorder
