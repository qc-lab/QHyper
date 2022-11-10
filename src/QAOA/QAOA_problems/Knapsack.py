import random
from collections import namedtuple

from .problem import Problem

 
Item = namedtuple('Item', "weight value")


class Knapsack:
    """Knapsack class"""

    def __init__(self, max_weight: int, N: int=0, max_item_value: int=10) -> None:
        self.items: list[Item] = []
        self.all_items: int = N
        self.max_weight: int = max_weight
        self.max_item_value: int = max_item_value
        self.generate_knapsack(N)

    def generate_knapsack(self, N: int) -> None:
        for _ in range(N):
            self.items.append(Item(
                    random.randint(1, self.max_weight), 
                    random.randint(1, self.max_item_value)
                )
            )

    def set_knapsack(self, items: list[tuple[int, int]]) -> None:
        self.items = [Item(weight, value) for weight, value in items]
        self.all_items = len(items)


class QAOA_Knapsack(Problem):
    """Class defining objective function and constraints for knapsack problem
    
    Attributes
    ----------
    objective_function : str
        objective function in SymPy syntax
    constraints : list[str]
        list of constraints in SymPy syntax
    wires : int
        number of qubits in the circuit, equals to number of items in knapsack + max weight of knapsack
    """

    def __init__(
        self, 
        knapsack: Knapsack, 
        # number_of_layers: int = 6, 
        # optimization_steps: int = 70,
        # optimizer: qml.GradientDescentOptimizer = None
        # optimizer: qml.GradientDescentOptimizer = qml.AdamOptimizer(
        #     stepsize=0.01,
        #     beta1=0.9,
        #     beta2=0.99
        # )
    ) -> None:
        """
        Parameters
        ----------
        knapsack : Knapsack
            knapsack object with available items and max weight
        """
        # self.number_of_layers = number_of_layers
        # self.optimization_steps = optimization_steps
        # self.optimizer = optimizer
        self.wires = knapsack.all_items + knapsack.max_weight
        # self.dev = qml.device("default.qubit", wires=self.wires)
        self._create_objective_function(knapsack)
        self._create_constraints(knapsack)

    # def _x(self, wire):
    #     return qml.Hamiltonian([0.5, -0.5], [qml.Identity(wire), qml.PauliZ(wire)])

    def _create_objective_function(self, knapsack: Knapsack) -> None:
        xs = [f"x{i}" for i in range(knapsack.all_items)]
        equation = "-("
        for i, x in enumerate(xs):
            equation += f"+{knapsack.items[i].value}*{x}"
        equation += ")"
        self.objective_function = equation

    def _create_constraints(self, knapsack: Knapsack) -> None:
        xs = [f"x{i}" for i in range(knapsack.all_items)]
        ys = [f"x{i}" for i in range(
                knapsack.all_items, knapsack.all_items + knapsack.max_weight)]
        constrains = []
        equation = f"(J"
        for y in ys:
            equation += f"-{y}"
        equation += f")**2"
        constrains.append(equation)
        equation = "("
        for i, y in enumerate(ys):
            equation += f"+{i+1}*{y}"
        for i, x in enumerate(xs):
            equation += f"-{knapsack.items[i].weight}*{x}"
        equation += ")**2"
        constrains.append(equation)
        self.constraints = constrains

    def get_score(self, result: str) -> float | None:
        """Method should return score of the provided outcome in bits. 
        
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
            if result[i + self.knapsack.all_items] == '1' and i+1 != weight:
                return None
        if weight != 0 and result[weight + self.knapsack.all_items - 1] != '1':
            return None

        return sum
