from dataclasses import dataclass
import random

from .problem import Problem


@dataclass
class Item:
    weight: int
    value: int


class Knapsack:
    def __init__(self, max_weight, N=0, max_item_value=10):
        self.items: list[Item] = []
        self.all_items = N
        self.max_weight = max_weight
        self.max_item_value = max_item_value
        self.generate_knapsack(N)

    def generate_knapsack(self, N):
        for _ in range(N):
            self.items.append(Item(
                    random.randint(1, self.max_weight), 
                    random.randint(1, self.max_item_value)
                )
            )

    def set_knapsack(self, items: list[tuple[int, int]]):
        self.items = [Item(weight, value) for weight, value in items]
        self.all_items = len(items)

    def calculate_value(self, items):
        weight = 0
        value = 0

        for i, is_taken in enumerate(items):
            if is_taken:
                weight += self.items[i].weight
                value += self.items[i].value
        return value if weight <= self.max_weight else -1

    def size(self):
        return len(self.items)


class QAOA_Knapsack(Problem):
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
        self.knapsack = knapsack
        # self.number_of_layers = number_of_layers
        # self.optimization_steps = optimization_steps
        # self.optimizer = optimizer
        self.wires = knapsack.all_items + knapsack.max_weight
        # self.dev = qml.device("default.qubit", wires=self.wires)
        self.create_objective_function(knapsack)
        self.create_constraints(knapsack)

    # def _x(self, wire):
    #     return qml.Hamiltonian([0.5, -0.5], [qml.Identity(wire), qml.PauliZ(wire)])

    def create_objective_function(self, knapsack: Knapsack):
        xs = [f"x{i}" for i in range(knapsack.all_items)]
        equation = "-("
        for i, x in enumerate(xs):
            equation += f"+{knapsack.items[i].value}*{x}"
        equation += ")"
        self.objective_function = equation

    def create_constraints(self, knapsack: Knapsack):
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

    def get_score(self, result) -> float | None:
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
