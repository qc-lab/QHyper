# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple, cast

import networkx as nx
import numpy as np
import sympy
from QHyper.problems.base import Problem
from QHyper.constraint import Constraint
from QHyper.polynomial import Polynomial
from sympy.core.expr import Expr
from QHyper.parser import from_sympy


@dataclass
class Network:
    graph: nx.Graph
    resolution: float = 1.0
    weight: str | None = "weight"
    community: list | None = None
    full_modularity_matrix: np.ndarray = field(init=False)
    generalized_modularity_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not self.community:
            self.community = [*range(self.graph.number_of_nodes())]
        (
            self.full_modularity_matrix,
            self.generalized_modularity_matrix,
        ) = self.calculate_modularity_matrix()

    def calculate_modularity_matrix(self) -> np.ndarray:
        adj_matrix: np.ndarray = nx.to_numpy_array(self.graph, weight=self.weight)
        in_degree_matrix: np.ndarray = adj_matrix.sum(axis=1)
        out_degree_matrix: np.ndarray = adj_matrix.sum(axis=0)
        m: int = np.sum(adj_matrix)
        full_modularity_matrix = (
            adj_matrix
            - self.resolution * np.outer(in_degree_matrix, out_degree_matrix) / m
        )

        B_bis = full_modularity_matrix[self.community, :]
        B_community = B_bis[:, self.community]
        B_i = np.sum(B_community, axis=1)
        B_j = np.sum(B_community.T, axis=1)
        delta = np.eye(len(self.community), dtype=np.int32)
        B_g = 0.5*( B_community + B_community.T ) - 0.5 * delta * (B_i + B_j)
        return full_modularity_matrix, B_g


class KarateClubNetwork(Network):
    def __init__(self, resolution: float = 1):
        super().__init__(nx.karate_club_graph(), resolution=resolution)


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
        super().__init__(nx.from_numpy_matrix(adj_matrix), resolution=resolution)


class CommunityDetectionProblem(Problem):
    """
    Problem class instance
    - objective function for network community detection

    Attributes
    ----------
    cases: int
        number of communities into which the graph shall be divided
        (default 2)
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

    def __init__(
        self,
        network_data: Network,
        communities: int = 2,
        one_hot_encoding: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        network_data: Network
            representation of a network with graph and modularity matrix
        communities: int
            number of communities into which the graph shall be divided
            (default 2)
        one_hot_encoding: bool
            decides if objective function should be encoded to one-hot
            values
        """
        self.G: nx.Graph = network_data.graph
        self.one_hot_encoding: bool = one_hot_encoding
        if one_hot_encoding:
            self.B: np.ndarray = network_data.full_modularity_matrix
        else:
            self.B: np.ndarray = network_data.generalized_modularity_matrix

        if communities < 1:
            raise Exception("Number of communities must be greater than or equal to 1")
        self.community = network_data.community
        self.cases: int = communities
        self.resolution: float = network_data.resolution

        if self.one_hot_encoding:
            self.variables: tuple[sympy.Symbol] = self._encode_discretes_to_one_hots()
            self._set_objective_function()
            self._set_one_hot_constraints(communities)
        else:
            self.variables: tuple[
                sympy.Symbol
            ] = self._get_discrete_variable_representation()
            self._set_objective_function()

    def _get_discrete_variable_representation(
        self,
    ) -> tuple[sympy.Symbol] | Any:
        return sympy.symbols(" ".join([f"x{i}" for i in range(len(self.community))]))

    def _set_objective_function(self) -> None:
        equation: dict[tuple[str, ...], float] = {}
        for i in range(len(self.B)):
            for j in range(len(self.B)):
                x_i, x_j = sympy.symbols(f"x{self.community[i]}"), sympy.symbols(
                    f"x{self.community[j]}"
                )
                if self.one_hot_encoding:
                    if i >= j:
                        continue
                    for case_val in range(self.cases):
                        s_i = str(self._encode_discrete_to_one_hot(x_i, case_val))
                        s_j = str(self._encode_discrete_to_one_hot(x_j, case_val))
                        equation[(s_i, s_j)], equation[(s_j, s_i)] = (
                            self.B[i, j],
                            self.B[j, i],
                        )
                else:
                    x_i, x_j = str(x_i), str(x_j)
                    equation[(x_i, x_j)] = self.B[i, j]

        equation = {key: -1 * val for key, val in equation.items()}

        self.objective_function = Polynomial(equation)

    def _encode_discrete_to_one_hot(
        self, discrete_variable: sympy.Symbol, case_value: int
    ) -> sympy.Symbol:
        discrete_id = int(str(discrete_variable)[1:])
        id = discrete_id * self.cases + case_value
        return sympy.symbols(f"s{id}")

    def _encode_discretes_to_one_hots(self) -> tuple[sympy.Symbol]:
        one_hots: tuple[sympy.Symbol] = sympy.symbols(
            " ".join(
                [
                    str(self._encode_discrete_to_one_hot(var, case_val))
                    for var in self._get_discrete_variable_representation()
                    for case_val in range(self.cases)
                ]
            )
        )
        return one_hots

    def iter_variables_cases(self) -> Iterable[Tuple[sympy.Symbol, ...]]:
        """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), ..."""
        return zip(*[iter(self.variables)] * self.cases)

    def _set_one_hot_constraints(self, communities: int) -> None:
        ONE_HOT_CONST = -1
        self.constraints: list[Constraint] = []
        # In the case of 1-to-1 mapping between discrete
        # and binary variable values no one-hot constraints
        if communities == ONE_HOT_CONST * -1:
            return

        dummies: Iterable[Tuple[sympy.Symbol, ...]]
        for dummies in self.iter_variables_cases():
            expression: Expr = cast(Expr, 0)
            dummy: sympy.Symbol
            for dummy in dummies:
                expression += dummy
            expression += ONE_HOT_CONST
            self.constraints.append(Constraint(from_sympy(expression)))

    def decode_solution(self, solution: dict) -> dict:
        ONE_HOT_VALUE = 1.0
        decoded_solution: dict = {}

        for variable, value in solution.items():
            id = int(variable[1:])
            if value == ONE_HOT_VALUE:
                case_value = id % self.cases
                variable_id = id // self.cases
                decoded_solution[variable_id] = case_value

        return self.sort_decoded_solution(decoded_solution)

    def sort_encoded_solution(self, encoded_solution: dict) -> dict:
        return {
            str(k): encoded_solution[str(k)]
            for k in self.variables
            if str(k) in encoded_solution
        }

    def sort_decoded_solution(self, decoded_solution: dict) -> dict:
        keyorder = [int(str(v)[1:]) for v in self.variables]
        return {k: decoded_solution[k] for k in keyorder if k in decoded_solution}
