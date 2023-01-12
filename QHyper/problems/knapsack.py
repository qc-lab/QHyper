import random
import sympy
from collections import namedtuple

from .problem import Problem
from QHyper.hyperparameter_gen.parser import Expression

Item = namedtuple('Item', "weight value")


class Knapsack:
    """Knapsack class

    Attributes
    ----------
    items: list[Item]
        list of items
    max_weight: int
        maximum weight of an item
    max_item_value: int
        maximum value of an item
    """

    def __init__(
        self, 
        max_weight: int, 
        max_item_value: int = 10, 
        items_amount: int = 0,
        items: list[tuple[int, int]] = []
    ) -> None:
        """
        Parameters
        ----------
        max_weight: int
            maximum weight of an item
        max_item_value: int
            maximum value of an item (default 10)
        items_amount: int
            items amount, used only for random knapsack (default 0)
        items: list[tuple[int, int]]
            set items in knapsack (default [])
        """
        self.items: list[Item] = []
        self.max_weight: int = max_weight
        self.max_item_value: int = max_item_value
        self.set_knapsack(items) if items else self.generate_knapsack(items_amount)

    def generate_knapsack(self, items_amount: int) -> None:
        for _ in range(items_amount):
            self.items.append(Item(
                random.randint(1, self.max_weight),
                random.randint(1, self.max_item_value)
            ))

    def set_knapsack(self, items: list[tuple[int, int]]) -> None:
        self.items = [Item(weight, value) for weight, value in items]
    
    def __len__(self):
        return len(self.items)


class KnapsackProblem(Problem):
    """Objective function and constraints for the knapsack problem
    
    Attributes
    ----------
    objective_function : str
        objective function in SymPy syntax
    constraints : list[str]
        list of constraints in SymPy syntax
    variables : int
        number of qubits in the circuit, equals to sum of the number of items in the knapsack the max weight of a knapsack
    """

    def __init__(
        self,
        max_weight: int, 
        max_item_value: int = 10, 
        items_amount: int = 0,
        items: list[tuple[int, int]] = []
    ) -> None:
        """
        Parameters
        ----------
        max_weight: int
            maximum weight of an item
        max_item_value: int
            maximum value of an item (default 10)
        items_amount: int
            items amount, used only for random knapsack (default 0)
        items: list[tuple[int, int]]
            set items in knapsack (default [])
        """
        self.knapsack = Knapsack(max_weight, max_item_value, items_amount, items)
        # self.variables = len(self.knapsack) + self.knapsack.max_weight
        self.variables = sympy.symbols(
            ' '.join([f'x{i}' for i in range(len(self.knapsack) + self.knapsack.max_weight)]))
        self._set_objective_function()
        self._set_constraints()

    def _set_objective_function(self) -> None:
        """
        Create the objective function defined in SymPy syntax
        """
        # xs = [f"x{i}" for i in range(len(self.knapsack))]
        equation = 0
        for i, x in enumerate(self.variables[:len(self.knapsack)]):
            equation += self.knapsack.items[i].value*x
        equation *= -1
        # equation += 
        self.objective_function = Expression(equation)

    def _set_constraints(self) -> None:
        """
        Create constraints defined in SymPy syntax
        """
        xs = [self.variables[i] for i in range(len(self.knapsack))]
        ys = [self.variables[i] for i in range(
            len(self.knapsack), len(self.knapsack) + self.knapsack.max_weight)]
        constrains = []
        equation = 1
        for y in ys:
            equation -= y
        equation = equation
        constrains.append(Expression(equation))
        equation = 0
        for i, y in enumerate(ys):
            equation += (i + 1)*y
        for i, x in enumerate(xs):
            equation += -(self.knapsack.items[i].weight)*x
        equation = equation
        constrains.append(Expression(equation))
        self.constraints = constrains

    def get_score(self, result: str) -> float | None:
        """Returns score of the provided outcome in bits
        
        Parameters
        ----------
        result : str
            outcome as a string of zeros and ones

        Returns
        -------
        float | None
            Returns sum of value of picked items if were picked correctly, else returns None
        """
        sum = 0
        weight = 0
        for i, item in enumerate(self.knapsack.items):
            if result[i] == '1':
                sum += item.value
                weight += item.weight
        if weight > self.knapsack.max_weight:
            return None

        for i in range(self.knapsack.max_weight):
            if result[i + len(self.knapsack)] == '1' and i + 1 != weight:
                return None
        if weight != 0 and result[weight + len(self.knapsack) - 1] != '1':
            return None

        return sum
