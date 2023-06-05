import sympy
from QHyper.problems.base import Problem
import networkx as nx
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import cast


class MaxCut:
    """
    MaxCut class

    Attributes
    ----------
    G: networkx graph representation of the problem
    """

    def __init__(self):
        self.G = nx.Graph()

    def add_edge(self, u, v):
        self.G.add_edge(u, v)

    def create_sample_graph(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)])


class MaxCutProblem(Problem):
    """
    Problem class instance
    - objective function (no constraints) for the max cut problem

    Attributes:
    ----------
    objective_function : str
        objective function in SymPy syntax
    constraints : list[str]
        empty list as there are no constraints in this problem
    variables : int
        number of qubits in the circuit, equals to number of nodes
        in the max cut graph
    """

    def __init__(self, max_cut: MaxCut, cuts_num: int = 2) -> None:
        """
        Parameters
        ----------
        max_cut: MaxCut
            instance of the MaxCut class to represent the problem
        cuts_num: int
            number of clusters into which the graph shall be divided
            (default 2)
        """
        self.max_cut_graph = max_cut.G
        self.num_cases = cuts_num
        self._set_variables()
        self._set_objective_function()
        self.constraints = []

    def _set_variables(self) -> None:
        """
        Set the variables in SymPy syntax
        """
        self.variables = sympy.symbols(
            " ".join([f"x{i}" for i in range(len(self.max_cut_graph.nodes))])
        )

    def _set_objective_function(self) -> None:
        """
        Create the objective function defined in SymPy syntax
        """
        # xs = [f"x{i}" for i in range(len(self.max_cut_graph.nodes))]
        equation: Expr = cast(Expr, 0)
        for u, v in self.max_cut_graph.edges:
            u_var, v_var = self.variables[u - 1], self.variables[v - 1]
            equation += u_var + v_var - 2 * u_var * v_var
        equation *= -1

        self.objective_function = Expression(equation)
