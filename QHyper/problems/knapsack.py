import random
from collections import namedtuple

from .problem import Problem

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

    def __init__(self, max_weight: int, max_item_value: int = 10, items_amount: int = 0) -> None:
        """
        Parameters
        ----------
        max_weight: int
            maximum weight of an item
        max_item_value: int
            maximum value of an item (default 10)
        items_amount: int
            items amount, used only for random knapsack (default 0)
        """
        self.items: list[Item] = []
        self.max_weight: int = max_weight
        self.max_item_value: int = max_item_value
        self.generate_knapsack(items_amount)

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
        knapsack: Knapsack
    ) -> None:
        """
        Parameters
        ----------
        knapsack : Knapsack
            knapsack object with available items and max weight
        """
        self.variables = len(knapsack) + knapsack.max_weight
        self.knapsack = knapsack
        self._set_objective_function(knapsack)
        self._set_constraints(knapsack)

    def _set_objective_function(self, knapsack: Knapsack) -> None:
        """
        Create the objective function defined in SymPy syntax

        Parameters
        ----------
        knapsack : Knapsack
            knapsack object with available items and max weight
        """
        xs = [f"x{i}" for i in range(len(knapsack))]
        equation = "-("
        for i, x in enumerate(xs):
            equation += f"+{knapsack.items[i].value}*{x}"
        equation += ")"
        self.objective_function = equation

    def _set_constraints(self, knapsack: Knapsack) -> None:
        """
        Create constraints defined in SymPy syntax

        Parameters
        ----------
        knapsack : Knapsack
               knapsack object with available items and max weight
        """
        xs = [f"x{i}" for i in range(len(knapsack))]
        ys = [f"x{i}" for i in range(
            len(knapsack), len(knapsack) + knapsack.max_weight)]
        constrains = []
        equation = f"(J"
        for y in ys:
            equation += f"-{y}"
        equation += f")**2"
        constrains.append(equation)
        equation = "("
        for i, y in enumerate(ys):
            equation += f"+{i + 1}*{y}"
        for i, x in enumerate(xs):
            equation += f"-{knapsack.items[i].weight}*{x}"
        equation += ")**2"
        constrains.append(equation)
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
