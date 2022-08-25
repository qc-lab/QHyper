from dataclasses import dataclass
import pennylane as qml
from pennylane import numpy as np
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
        number_of_layers: int = 6, 
        optimization_steps: int = 70,
        optimizer: qml.GradientDescentOptimizer = qml.AdagradOptimizer()
    ) -> None:
        self.knapsack = knapsack
        self.number_of_layers = number_of_layers
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer
        self.wires = knapsack.all_items + knapsack.max_weight
        self.dev = qml.device("default.qubit", wires=self.wires)

    def _x(self, wire):
        return qml.Hamiltonian([0.5, -0.5], [qml.Identity(wire), qml.PauliZ(wire)])

    def _create_cost_operator(self, parameters):
        A, B = parameters
        hamiltonian = qml.Identity(0) #  remove
        xs = range(0, self.knapsack.all_items)
        ys = list(range(self.knapsack.all_items, self.knapsack.all_items + self.knapsack.max_weight))
        
        for y in ys:
            hamiltonian -= A * self._x(y)
        
        for y in range(1, self.knapsack.max_weight+1):
            hamiltonian += A * y**2 * self._x(ys[y-1])
        
        for _x in xs:
            hamiltonian += A * self.knapsack.items[_x].weight**2 * self._x(_x)
        
        for y1 in range(1, self.knapsack.max_weight+1):
            for y2 in range(y1 + 1, self.knapsack.max_weight+1):
                hamiltonian += A * 2 * y1 * y2 * (self._x(ys[y1-1]) @ self._x(ys[y2-1]))
        
        for _x1 in xs:
            for _x2 in range(_x1 + 1, self.knapsack.all_items):
                hamiltonian += (
                    A * self.knapsack.items[_x1].weight
                    * self.knapsack.items[_x2].weight 
                    * (self._x(_x1) @ self._x(_x2))
                )

        for y1 in range(1, self.knapsack.max_weight+1):
            for _x1 in xs:
                hamiltonian -= (
                    A * y1 * self.knapsack.items[_x1].weight *
                    (self._x(ys[y1-1]) @ self._x(_x1))
                )
            
        for _x in xs:
            hamiltonian -= B * self.knapsack.items[_x].value * self._x(_x)

        return hamiltonian

    def _get_value(self, result):
        sum = 0
        weight = 0
        for i, item in enumerate(self.knapsack.items):
            if result[i] == '1':
                sum += item.value
                weight += item.weight
        if weight > self.knapsack.max_weight:
            return -1

        for i in range(self.knapsack.max_weight):
            if result[i + self.knapsack.all_items] == '1' and i+1 != weight:
                return -1
        if weight != 0 and result[weight + self.knapsack.all_items - 1] != '1':
            return -1

        return sum

    def _check_results(self, probs):
        get_bin = lambda x: format(x, 'b').zfill(self.wires)
        
        result_dict = {key: float(val) for key, val in enumerate(probs)}
        result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
        score = 0
        for key, val in result_dict.items():
            binary_rep = get_bin(key)
            if (value:=self._get_value(binary_rep)) == -1:
                score += 0 #val*5
            else:
                score -= 2*val*value
        return score

    def print_results(self, parameters):
        get_bin = lambda x: format(x, 'b').zfill(self.wires)
        probs = self._run_learning(parameters)
        result_dict = {key: float(val) for key, val in enumerate(probs)}
        result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
        for key, val in result_dict.items():
            binary_rep = get_bin(key)

            print(
                f"Key: {get_bin(key)} with probability {val}   "
                f"| correct: {'True, value: '+str(self._get_value(binary_rep)) if self._get_value(binary_rep) != -1 else 'False'}"
            )
