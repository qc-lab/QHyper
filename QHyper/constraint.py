import uuid
from enum import Enum

from QHyper.polynomial import Polynomial


class MethodsForInequalities(Enum):
    SLACKS_LOG_2 = 0
    UNBALANCED_PENALIZATION = 1


SLACKS_LOG_2 = MethodsForInequalities.SLACKS_LOG_2
UNBALANCED_PENALIZATION = MethodsForInequalities.UNBALANCED_PENALIZATION


class Operator(Enum):
    EQ = "=="
    GE = ">="
    LE = "<="


class Constraint:
    """
    A class to represent a constraint.
    """
    def __init__(
        self,
        lhs: Polynomial,
        rhs: Polynomial = Polynomial(0),
        operator: Operator = Operator.EQ,
        method_for_inequalities: MethodsForInequalities | None = None,
        label: str = "",
        group: int = -1,
    ) -> None:
        """
            For now, we assume that the constraint is in
            the form of: sum of something <= number
        """

        self.lhs: Polynomial = lhs
        self.rhs: Polynomial = rhs
        self.operator: Operator = operator

        if operator != Operator.EQ and method_for_inequalities is None:
            raise Exception(
                "Method for inequalities must be "
                "provided when operator is not =="
            )
        self.method_for_inequalities = method_for_inequalities
        self._set_label(label)
        self.group = group

    def _set_label(self, label: str) -> None:
        self.label = label or f"s{uuid.uuid4().hex}"

    def __repr__(self) -> str:
        return f"{self.lhs} {self.operator.value} {self.rhs}"

    def get_variables(self) -> set[str]:
        return self.lhs.get_variables() | self.rhs.get_variables()


def get_number_of_constraints(constraints: list[Constraint]) -> int:
    """Returns the number of unique groups in the constraints list.
    """

    counter = 0
    visited = set()

    for c in constraints:
        if c.group == -1:
            counter += 1
        elif c.group not in visited:
            visited.add(c.group)
            counter += 1
    return counter
