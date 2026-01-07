"""This module contains the Constraint class which is used to represent a, well, 
constraint. It is used in the QUBO formulation to define the left and right
hand sides of the constraints, and the operator.

.. rubric:: Main class

.. autosummary::
    :toctree: generated

    Constraint  -- implementation of the constraint.

.. rubric:: Enum Classes

.. autoclass:: MethodsForInequalities


.. rubric:: Functions

.. autofunction:: get_number_of_constraints

"""

import uuid
from enum import Enum

from QHyper.polynomial import Polynomial, PolynomialType


class MethodsForInequalities(Enum):
    """Enum class with different methods for handling inequalities.

    There are two available methods for handling inequalities:
    .. list-table::

        * - SLACKS_LOG_2
          - UNBALANCED_PENALIZATION 
        * - the method uses slack variables in number of log2(n)
          - the method uses unbalanced penalization
    """
    SLACKS_LOG_2 = 0
    UNBALANCED_PENALIZATION = 1


SLACKS_LOG_2 = MethodsForInequalities.SLACKS_LOG_2
UNBALANCED_PENALIZATION = MethodsForInequalities.UNBALANCED_PENALIZATION


class Operator(Enum):
    """Enum class with different operators.
    """
    EQ = "=="
    GE = ">="
    LE = "<="


class Constraint:
    """
    A class representing the constraint.

    Attributes
    ----------
    lhs : PolynomialType
        The left-hand side of the constraint.
    rhs : PolynomialType, default 0
        The right-hand side of the constraint.
    operator : Operator, default Operator.EQ
        The operator of the constraint. It can be ==, >=, <=.
    method_for_inequalities : MethodsForInequalities, optional
        The method to be used for inequalities. It can be SLACKS_LOG_2 or
        UNBALANCED_PENALIZATION. It is required when the operator is not ==.
    label : str, optional
        The label of the constraint. If not provided, it will be set to a
        random string.
    group : int, default -1
        The group of the constraint. It is used to group constraints together.
        Example use is assigning same weight to the constraints with the same
        group when creating qubo.
    """

    def __init__(
        self,
        lhs: PolynomialType,
        rhs: PolynomialType = 0,
        operator: Operator = Operator.EQ,
        method_for_inequalities: MethodsForInequalities | None = None,
        label: str = "",
        group: int = -1,
    ) -> None:
        self.lhs = lhs if isinstance(lhs, Polynomial) else Polynomial(lhs)
        self.rhs = rhs if isinstance(rhs, Polynomial) else Polynomial(rhs)
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
