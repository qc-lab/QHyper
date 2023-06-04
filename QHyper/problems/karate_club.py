import sympy
from QHyper.problems.base import Problem
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import cast
from QHyper.problems.base import Problem
import networkx as nx


class KarateClubProblem(Problem):
    def __init__(self, num_clusters=2):
        self.num_cases = num_clusters
        self.G = nx.karate_club_graph()
        self.B = nx.modularity_matrix(self.G)
        self._set_variables()
        self._set_objective_function()
        self.constraints = []

    def _set_variables(self) -> None:
        self.variables = sympy.symbols(
            " ".join([f"x{i}" for i in range(len(self.G.nodes))])
        )

    def _set_objective_function(self) -> None:
        """
        Create the objective function on defined in SymPy syntax
        """
        # xs = [f"x{i}" for i in range(len(self.G.nodes))]
        equation: Expr = cast(Expr, 0)
        for i in self.G.nodes():
            for j in range(i + 1, len(self.G.nodes)):
                u_var, v_var = self.variables[i], self.variables[j]
                equation += u_var * v_var * self.B[i, j]
        equation *= -1

        self.objective_function = Expression(equation)
