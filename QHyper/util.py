import ast
import uuid
from _ast import Compare
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Union, cast

import sympy

VARIABLES = Union[tuple[()], tuple[str], tuple[str, str], tuple[str, ...]]
QUBO = dict[VARIABLES, float]


@dataclass
class Component:
    """Represents a mathematical component with variables separated from the numerical coefficient.
    For example, Component(['x1', 'x2'], 2) corresponds to 2 * 'x1' * 'x2'.

    Attributes:
    ----------
    variables : A list of variables.
    coefficient : The coefficient.
    """

    variables: list[str] = field(default_factory=list)
    coefficient: int = field(default=1)

    def add_variable(self, variable: str) -> None:
        self.variables.append(variable)

    def set_coefficient(self, new_coefficient: int) -> None:
        self.coefficient = new_coefficient


class Parser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.polynomial_as_dict: QUBO = {}

    def visit_Expr(self, node: ast.Expr) -> None:
        visited = self.visit(node.value)

        if isinstance(visited, Component):
            self.polynomial_as_dict[tuple(visited.variables)] = visited.coefficient
        elif isinstance(visited, list):
            for component in visited:
                self.polynomial_as_dict[
                    tuple(component.variables)
                ] = component.coefficient
        else:
            raise Exception("TBD")

    def visit_Constant(self, node: ast.Constant) -> Component:
        return Component(coefficient=node.value)
        # return Component([], node.value)

    def visit_Name(self, node: ast.Name) -> Component:
        return Component(variables=[node.id])
        # return Component([node.id], 1)

    def visit_BinOp(self, node: ast.BinOp) -> Component | list[Component]:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return self.combine_components(lhs, rhs)

        if isinstance(node.op, ast.Sub):
            return self.combine_components(lhs, rhs, is_subtraction=True)

        if isinstance(node.op, ast.Mult):
            return Component(
                lhs.variables + rhs.variables, lhs.coefficient * rhs.coefficient
            )

        if isinstance(node.op, ast.Pow):
            return Component(
                [node.left.id for _ in range(node.right.value)], 1
            )  # todo is it really 1?

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:  # todo what it really returns?
        lhs = self.visit(node.operand)

        if isinstance(node.op, ast.USub):
            previous_coefficient = lhs.coefficient
            lhs.set_coefficient(-1 * previous_coefficient)
            return lhs

        if isinstance(node.op, ast.UAdd):  # TODO - check if it is needed
            return lhs

    @staticmethod
    def combine_components(
        left: Component | list[Component],
        right: Component | list[Component],
        is_subtraction: bool = False,
    ) -> list[Component]:
        if is_subtraction:
            right.set_coefficient(-1 * right.coefficient)

        if isinstance(left, Component) and isinstance(right, Component):
            return [left, right]
        elif isinstance(left, list) and isinstance(right, Component):
            return left + [right]
        elif isinstance(left, list) and isinstance(right, list):
            return left + right
        else:
            raise Exception("TBD")


class MethodsForInequalities(Enum):  # todo penalization method
    SLACKS_LOG_2 = 0
    UNBALANCED_PENALIZATION = 1


class Operator(Enum):  # todo comparison operator
    EQ = "=="
    GE = ">="
    LE = "<="


class Expression:
    def __init__(self, equation: sympy.core.Expr | dict) -> None:
        self._set_dictionary(equation)

    def _set_dictionary(self, equation: sympy.core.Expr | dict) -> None:
        if isinstance(equation, dict):
            self.dictionary = equation
        elif isinstance(equation, sympy.core.Expr):
            parser = Parser()
            ast_tree = ast.parse(
                str(sympy.expand(equation))
            )  # type: ignore[no-untyped-call]
            parser.visit(ast_tree)
            self.dictionary = parser.polynomial_as_dict
        else:
            raise Exception(
                "Expression equation must be an instance of "
                "sympy.core.Expr or dict, got of type: "
                f"{type(equation)} instead"
            )

    def __repr__(self) -> str:
        return str(self.dictionary)

    def as_polynomial(self) -> str:
        if self.polynomial is not None:
            return str(self.polynomial)
        else:
            polynomial = str()
            for k in self.dictionary:
                if self.dictionary[k] < 0:
                    polynomial += "- "
                polynomial += str(abs(self.dictionary[k])) + "*"
                polynomial += "*".join(k)
                polynomial += " "
            return polynomial.rstrip()


class Constraint:
    def __init__(
        self,
        lhs: QUBO | Expression,
        rhs: int | float,
        operator: Operator = Operator.EQ,
        method_for_inequalities: MethodsForInequalities = MethodsForInequalities.SLACKS_LOG_2,
        label: str = "",
    ) -> None:
        """For now, we assume that the constraint is in the form of: sum of something <= number"""
        self._set_lhs(lhs)
        self.rhs: int | float = rhs
        self.operator: Operator = operator  # for now it is only LE: number <= some polynomial without constants
        self._set_label(label)
        self.method_for_inequalities = method_for_inequalities

    def setup(
        self,
    ):
        ...

    def _set_lhs(self, lhs):
        if isinstance(lhs, dict):
            self.lhs: QUBO = lhs
        elif isinstance(lhs, Expression):
            # todo expression to dict
            print("in here dictionary ", lhs.dictionary)
            self.lhs: QUBO = lhs.dictionary  # todo

    def _set_label(self, label: str) -> None:
        if label == "":
            self.label = f"s{uuid.uuid4().hex}"
        self.label = label

    def __repr__(self) -> str:
        return f"{self.lhs} {self.operator.value} {self.rhs}"


def weighted_avg_evaluation(
    results: dict[str, float],
    score_function: Callable[[str, float], float],
    penalty: float = 0,
    limit_results: int | None = None,
    normalize: bool = True,
) -> float:
    score: float = 0

    sorted_results = sort_solver_results(results, limit_results)
    if normalize:
        scaler = 1 / sum([v for v in sorted_results.values()])
    else:
        scaler = 1

    for result, prob in sorted_results.items():
        score += scaler * prob * score_function(result, penalty)
    return score


def sort_solver_results(
    results: dict[str, float],
    limit_results: int | None = None,
) -> dict[str, float]:
    limit_results = limit_results or len(results)
    return {
        k: v
        for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)[
            :limit_results
        ]
    }


def add_evaluation_to_results(
    results: dict[str, float],
    score_function: Callable[[str, float], float],
    penalty: float = 1,
) -> dict[str, tuple[float, float]]:
    """
    Parameters
    ----------
    results : dict[str, float]
        dictionary of results
    score_function : Callable[[str, float], float]
        function that receives result and penalty and returns score, probably
        will be passed from Problem.get_score

    Returns
    -------
    dict[str, tuple[float, float]]
        dictionary of results with scores
    """

    return {k: (v, score_function(k, penalty)) for k, v in results.items()}
