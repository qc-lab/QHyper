# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import cast

import numpy as np
import sympy

from QHyper.parser import from_sympy

from QHyper.problems.base import Problem


class MaxCutProblem(Problem):
    """MaxCut problem

    Parameters
    ----------
    edges : list[tuple[int, int]]
        List of edges in the graph

    Attributes
    ----------
    objective_function: Polynomial
        Objective_function represented as a Polynomial
    constraints : list[Polynomial]
        For MaxCut problem, there are no constraints, so it's empty list
    edges : list[tuple[int, int]]
        List of edges in the graph
    """

    def __init__(self, edges: list[tuple[int, int]]) -> None:
        self.edges = edges
        self.variables = sympy.symbols(
            " ".join(
                [f"x{i}" for i in range(max(v for edge in edges
                                            for v in edge))]
            )
        )

        self._set_objective_function()
        self.constraints = []

    def _set_objective_function(self) -> None:
        equation = cast(sympy.Expr, 0)

        for e in self.edges:
            x_i, x_j = self.variables[e[0] - 1], self.variables[e[1] - 1]
            equation -= x_i * (1 - x_j) + x_j * (1 - x_i)

        self.objective_function = from_sympy(equation)

    def get_score(self, result: np.record, penalty: float = 0) -> float:
        sum = 0

        for e in self.edges:
            x_i, x_j = int(result[f"x{e[0] - 1}"]), int(result[f"x{e[1] - 1}"])
            sum += x_i * (1 - x_j) + x_j * (1 - x_i)

        return sum
