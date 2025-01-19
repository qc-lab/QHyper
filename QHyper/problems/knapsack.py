# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


"""
.. currentmodule:: QHyper.problems.knapsack
"""

import random
import sympy
import numpy as np
from collections import namedtuple

from QHyper.polynomial import Polynomial
from QHyper.problems.base import Problem
from QHyper.constraint import Constraint

Item = namedtuple('Item', "weight value")


class Knapsack:
    """Knapsack class

    Attributes
    ----------
    max_weight: int
        maximum weight of an item
    max_item_value: int
        maximum value of an item
    items: list[Item]
        list of items
    """

    def __init__(
        self,
        max_weight: int,
        max_item_value: int,
        items_amount: int,
        item_weights: list[int],
        item_values: list[int]
    ) -> None:
        """
        Parameters
        ----------
        max_weight: int
            maximum weight of an item
        max_item_value: int
            maximum value of an item
        items_amount: int
            items amount, used only for random knapsack
        items: list[tuple[int, int]]
            set items in knapsack
        """
        self.items: list[Item] = []
        self.max_weight: int = max_weight
        self.max_item_value: int = max_item_value
        if item_weights and item_values:
            if len(item_weights) != len(item_values):
                raise ValueError(
                    "Weights and values must have the same length")
            self.set_knapsack(item_weights, item_values)
        else:
            if items_amount < 1:
                raise ValueError(
                    "Cannot create knapsack with less than one item")
            self.generate_knapsack(items_amount)

    def generate_knapsack(self, items_amount: int) -> None:
        for _ in range(items_amount):
            self.items.append(Item(
                random.randint(1, self.max_weight),
                random.randint(1, self.max_item_value)
            ))

    def set_knapsack(self, weights: list[int], values: list[int]
                     ) -> None:
        self.items = [Item(weight, value)
                      for weight, value in zip(weights, values)]

    def __len__(self) -> int:
        return len(self.items)


class KnapsackProblem(Problem):
    """Objective function and constraints for the knapsack problem

    Parameters
    ----------
    max_weight: int
        maximum weight of an item
    max_item_value: int, default 10
        maximum value of an item
    items_amount: int, optional
        items amount, used only for random knapsack. If not provided,
        then item_weights and item_values must be specified
    item_weights: list[int], optional
        list of items weights
    item_values: list[int], optional
        list of items values

    Attributes
    ----------
    objective_function : Polynomial
        objective function in SymPy syntax wrapped in Expression class
    constraints : list[Polynomial]
        list of constraints in SymPy syntax wrapped in Expression class
    knapsack: :py:class:`Knapsack`
        Knapsack instance
    """

    def __init__(
        self,
        max_weight: int,
        max_item_value: int = 10,
        items_amount: int = 1,
        item_weights: list[int] = [],
        item_values: list[int] = []
    ) -> None:
        self.knapsack = Knapsack(max_weight, max_item_value,
                                 items_amount, item_weights, item_values)
        self.variables: tuple[sympy.Symbol] = sympy.symbols(' '.join(
            [f'x{i}' for i
             in range(len(self.knapsack) + self.knapsack.max_weight)]
        ))
        self._set_objective_function()
        self._set_constraints()

    def _set_objective_function(self) -> None:
        """
        Create the objective function items on defined in SymPy syntax
        """
        equation = Polynomial(0)
        for i in range(len(self.knapsack.items)):
            equation += Polynomial({(f"x{i}", ): self.knapsack.items[i].value})
        equation = -equation

        self.objective_function = equation

    def _set_constraints(self) -> None:
        """
        Create constraints defined in SymPy syntax
        """
        self.constraints: list[Constraint] = []
        equation = Polynomial(1)
        for i in range(self.knapsack.max_weight):
            equation -= Polynomial({(f'x{i+len(self.knapsack)}',): 1})
        self.constraints.append(Constraint(equation))
        equation = Polynomial(0)
        for i in range(self.knapsack.max_weight):
            equation += Polynomial({(f'x{i+len(self.knapsack)}', ): (i+1)})
        for i in range(len(self.knapsack.items)):
            equation -= Polynomial({(f"x{i}",): self.knapsack.items[i].weight})
        self.constraints.append(Constraint(equation))

    def get_score(self, result: np.record, penalty: float = 0) -> float:
        """Returns score for the provided numpy recor

        Parameters
        ----------
        result : np.record
            Outcome as a numpy record with variables as keys and their values.
            Dtype is list of tuples with variable name and its value (0 or 1)
            and tuple ('probability', <float>).
        penalty : float, default 0
            Penalty for the constraint violation

        Returns
        -------
        float
            Returns negated sum of value of picked items or 0 if knapsack
            isn't correct
        """
        sum = 0
        weight = 0
        for i, item in enumerate(self.knapsack.items):
            if result[f'x{i}'] == 1:
                sum += item.value
                weight += item.weight
        if weight > self.knapsack.max_weight:
            return penalty

        for i in range(self.knapsack.max_weight):
            if result[f'x{i + len(self.knapsack)}'] == 1 and i + 1 != weight:
                return penalty
        if weight != 0 and result[f'x{weight + len(self.knapsack) - 1}'] != 1:
            return penalty

        return -sum
