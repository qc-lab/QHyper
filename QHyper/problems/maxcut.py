# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import numpy as np
import sympy

from QHyper.polynomial import Polynomial

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
                                            for v in edge) + 1)]
            )
        )

        self._set_objective_function()
        self.constraints = []

    def _set_objective_function(self) -> None:
        equation = Polynomial(0)

        for e in self.edges:
            
            x_i = f"x{e[0]}"
            x_j = f"x{e[1]}"
            equation -= Polynomial({(x_i,): 1, (x_j,): 1, (x_i, x_j): -2})
        self.objective_function = equation

    def get_score(self, result: np.record, penalty: float = 0) -> float:
        sum = 0
        for e in self.edges:
            x_i, x_j = int(result[f"x{e[0]}"]), int(result[f"x{e[1]}"])
            sum += x_i * (1 - x_j) + x_j * (1 - x_i)
        return -sum
