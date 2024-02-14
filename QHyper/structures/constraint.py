import ast
import uuid
from enum import Enum

import sympy


class MethodsForInequalities(Enum):  # todo penalization method
    SLACKS_LOG_2 = 0
    UNBALANCED_PENALIZATION = 1


class Operator(Enum):  # todo comparison operator
    EQ = "=="
    GE = ">="
    LE = "<="


class Expression:
    def __init__(self, equation: dict) -> None:
        if isinstance(equation, dict):
            self.dictionary = equation
        else:
            raise Exception(
                "Expression equation must be an instance of "
                "sympy.core.Expr or dict, got of type: "
                f"{type(equation)} instead"
            )

    def __repr__(self) -> str:
        return str(self.dictionary)

    def to_sympy(self) -> str:
        polynomial = str()
        for k in self.dictionary:
            if self.dictionary[k] < 0:
                polynomial += "- "
            polynomial += str(abs(self.dictionary[k])) + "*"
            polynomial += "*".join(k)
            polynomial += " "
        return polynomial.rstrip()

    @staticmethod
    def from_sympy(equation: sympy.core.Expr) -> "Expression":
        parser = Parser()
        ast_tree = ast.parse(str(sympy.expand(equation)))
        parser.visit(ast_tree)
        return Expression(parser.polynomial_as_dict)

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
            self.lhs: QUBO = lhs.dictionary  # todo

    def _set_label(self, label: str) -> None:
        if label == "":
            self.label = f"s{uuid.uuid4().hex}"
        self.label = label

    def __repr__(self) -> str:
        return f"{self.lhs} {self.operator.value} {self.rhs}"


