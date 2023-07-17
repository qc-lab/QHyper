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


class ObjFunFormula(Enum):
    SYMPY_EXPR = 1
    DICT = 2


@dataclass
class Network:
    graph: nx.Graph
    modularity_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
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

    def __init__(
        self,
        network_data: Network,
        N_communities: int = 2,
        obj_func_formula: ObjFunFormula = ObjFunFormula.SYMPY_EXPR,
    ) -> None:
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
        self.variables: tuple[sympy.Symbol] = sympy.symbols(
            " ".join([f"x{i}" for i in range(len(self.G.nodes))])
        )
        self.constraints = []
        self.dummy_coefficients: dict = self._get_dummy_coefficients()

        self._set_objective_function()
        self._set_one_hot_constraints()

        self.variables: tuple[sympy.Symbol] = self._get_variables_from_dummy_coefficients()

    def _get_dummy_coefficients(self) -> dict:
        dummy_coefficients: dict = {
            var: sympy.symbols(
                " ".join(
                    [
                        f"s{k}"
                        for k in range(
                            i * self.N_cases, (i + 1) * self.N_cases
                        )
                    ]
                )
            )
            for i, var in enumerate(self.variables)
        }
        return dummy_coefficients
    
    def _get_variables_from_dummy_coefficients(self) -> tuple[sympy.Symbol]:
        # print(self.dummy_coefficients.values())
        variables: tuple[sympy.Symbol] = sympy.symbols(
            " ".join([f"{str(var_name)}" for _, v in self.dummy_coefficients.items() for var_name in v])
        )
        print(f"variables: {variables}")
        return variables

    def _set_objective_function(self) -> None:
        # xs = [f"x{i}" for i in range(len(self.G.nodes))]
        equation: dict[VARIABLES, float] = {}

        for i in self.G.nodes:
            for j in range(i + 1, len(self.G.nodes)):
                u_var, v_var = self.variables[i], self.variables[j]
                for case in range(self.N_cases):
                    u_var_dummy = str(self.dummy_coefficients[u_var][case])
                    v_var_dummy = str(self.dummy_coefficients[v_var][case])
                    equation[(u_var_dummy, v_var_dummy)] = self.B[i, j]

        equation = {key: -1 * val for key, val in equation.items()}

        self.objective_function = Expression(equation)
        # self._set_one_hot_constraints()

    def _set_one_hot_constraints(self) -> None:
        self.constraints: list[Expression] = []
        ONE_HOT_CONST = -1

        for _, dummies in self.dummy_coefficients.items():
            expression: Expr = cast(Expr, 0)
            for dummy in dummies:
                expression += dummy
            expression += ONE_HOT_CONST
            self.constraints.append(Expression(expression))

    def decode_dummies_solution(self, solution: dict) -> dict:
        ONE_HOT_VALUE = 1.0
        decoded_solution = {}

        for variable, value in solution.items():
            _, id = variable[0], int(variable[len("s"):])
            if value == ONE_HOT_VALUE:
                case_value = id % self.N_cases
                variable_id = id // self.N_cases
                decoded_solution[variable_id] = case_value

        return decoded_solution
