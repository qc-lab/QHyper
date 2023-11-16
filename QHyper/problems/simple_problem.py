from typing import cast

import sympy
from QHyper.util import Expression, Constraint, Operator, MethodsForInequalities

from .base import Problem

# import ast
# import astpretty


class SimpleProblem(Problem):
    def __init__(self) -> None:
        self.num_variables = 2
        self.variables = sympy.symbols(
            " ".join(
                [f"x{i}" for i in range(self.num_variables)]
            )
        )

        self._set_objective_function()
        self._set_constraints()
        self.method_for_inequalities = MethodsForInequalities.SLACKS_LOG_2
        # self.method_for_inequalities = MethodsForInequalities.UNBALANCED_PENALIZATION

    def _set_objective_function(self) -> None:
        equation = cast(sympy.Expr, 0)

        equation =  (5 * self.variables[0] + 2 * self.variables[1] + self.variables[0] * self.variables[1])

        self.objective_function: Expression = Expression(equation)


    def _set_constraints(self) -> None:
        self.constraints: list[Constraint] = []

        constraint_eq_rhs = {('x0',): 1, ('x1', ): 1, (): -1}
        constraint_eq = Constraint(constraint_eq_rhs, 0, Operator.EQ)
        
        constraint_le_rhs = {('x0',) : 5, ('x1', ): 2, ('x0', 'x1'): 1}
        constraint_le = Constraint(constraint_le_rhs, 5, Operator.LE)
        
        # self.constraints.append(constraint_eq)
        self.constraints.append(constraint_le)

    def get_score(self, result: str, penalty: float = 0) -> float:
        print("get score result", result)
        return -(5*int(result[0]) + 3 * int(result[1]))
