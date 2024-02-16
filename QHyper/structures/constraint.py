import uuid
from enum import Enum

from QHyper.structures.polynomial import Polynomial


class MethodsForInequalities(Enum):  # todo penalization method
    SLACKS_LOG_2 = 0
    UNBALANCED_PENALIZATION = 1


class Operator(Enum):  # todo comparison operator
    EQ = "=="
    GE = ">="
    LE = "<="


class Constraint:
    def __init__(
        self,
        lhs: Polynomial,
        rhs: Polynomial,
        operator: Operator = Operator.EQ,
        method_for_inequalities: MethodsForInequalities | None = None,
        label: str = "",
    ) -> None:
        """For now, we assume that the constraint is in the form of: sum of something <= number"""
        self.lhs: Polynomial = lhs
        self.rhs: Polynomial = rhs
        self.operator: Operator = operator

        if operator != Operator.EQ and method_for_inequalities is None:
            raise Exception(
                f"Method for inequalities must be provided when operator is not =="
            )
        self.method_for_inequalities = method_for_inequalities
        self._set_label(label)

    def _set_label(self, label: str) -> None:
        if label == "":
            self.label = f"s{uuid.uuid4().hex}"
        self.label = label

    def __repr__(self) -> str:
        return f"{self.lhs} {self.operator.value} {self.rhs}"
