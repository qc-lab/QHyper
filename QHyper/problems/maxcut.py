import random
import sympy
from collections import namedtuple

from typing import cast

from .base import Problem
from QHyper.hyperparameter_gen.parser import Expression


class MaxCutProblem(Problem):
    def __init__(self, edges: list[tuple[int, int]]) -> None:
        self.edges = edges
        self.variables = sympy.symbols(' '.join(
            [f'x{i}' for i in range(max(v for edge in edges for v in edge) + 1)]
        ))

        self._set_objective_function()

    def _set_objective_function(self) -> None:
        equation = cast(sympy.Expr, 0)

        for e in self.edges:
            x_i, x_j = self.variables[e[0]], self.variables[e[1]]
            equation -= x_i * (1 - x_j) + x_j * (1 - x_i)

        self.objective_function: Expression = Expression(equation)

    def get_score(self, result: str) -> float | None:
        sum = 0

        for e in self.edges:
            x_i, x_j = int(result[e[0]]), int(result[e[1]])
            sum += x_i * (1 - x_j) + x_j * (1 - x_i)

        return sum
