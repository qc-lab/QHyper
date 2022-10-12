import pennylane as qml
import numpy as np

from .parser import parse_hamiltonian
from ..solver import Solver
from ...QAOA_problems.problem import Problem
from ...optimizers.optimizer import Optimizer, HyperparametersOptimizer


class PennyLaneQAOA(Solver):
    def __init__(self, **kwargs) -> None:
        self.problem: Problem = kwargs.get("problem")
        self.angles: list[float] = kwargs.get("angles", None)

        self.optimizer: Optimizer = kwargs.get("optimizer", None)
        self.layers: int = kwargs.get("layers", 3)
        self.mixer: str = kwargs.get("mixer", "X")
        self.weights: list[float] = kwargs.get("weights", None)
        self.hyperoptimizer: HyperparametersOptimizer = kwargs.get("hyperoptimizer", None)
        self.backend: str = kwargs.get("backend", None)

        if self.backend is None:
            self.dev = qml.device("default.qubit", wires=self.problem.wires)

    def create_cost_operator(self, weights) -> qml.Hamiltonian:
        ingredients = [self.problem.objective_function] + self.problem.constraints
        cost_operator = ""
        for weight, ingredient in zip(weights, ingredients):
            cost_operator += f"+{weight}*({ingredient})"
        return parse_hamiltonian(cost_operator)

    def _hadamard_layer(self):
        for i in range(self.problem.wires):
            qml.Hadamard(i)
    
    def create_mixing_hamitonian(self):
        hamiltonian = qml.Hamiltonian([], [])
        for i in range(self.problem.wires):
            hamiltonian += qml.Hamiltonian([1/2], [qml.PauliX(i)])
        return hamiltonian

    def circuit(self, params, cost_operator: qml.Hamiltonian):
        def qaoa_layer(gamma, beta):
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(beta, self.create_mixing_hamitonian()) 

        self._hadamard_layer()
        qml.layer(qaoa_layer, self.layers, params[0], params[1]) 

    def get_expval_func(self, weights):
        cost_operator = self.create_cost_operator(weights)
        @qml.qnode(self.dev)
        def cost_function(params):
            self.circuit(params, cost_operator)
            x = qml.expval(cost_operator)
            return x
        
        return cost_function

    def get_probs_func(self, weights):
        cost_operator = self.create_cost_operator(weights)
        @qml.qnode(self.dev)
        def probability_circuit(params):
            self.circuit(params, cost_operator)
            return qml.probs(wires=range(self.problem.wires))

        return probability_circuit

    def check_results(self, probs):
        to_bin = lambda x: format(x, 'b').zfill(self.problem.wires)
        
        results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
        results_by_probabilites = dict(
            sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
        score = 0
        for result, prob in results_by_probabilites.items():
            if (value:=self.problem.get_score(to_bin(result))) == -1:
                score += 0 # experiments?
            else:
                score -= prob*value
        return score * 100
    
    def print_results(self, probs):
        to_bin = lambda x: format(x, 'b').zfill(self.problem.wires)
        results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
        results_by_probabilites = dict(
            sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
        for result, prob in results_by_probabilites.items():
            # binary_rep = to_bin(key)
            value = self.problem.get_score(to_bin(result))
            print(
                f"Key: {to_bin(result)} with probability {prob}   "
                f"| correct: {'True, value: '+str(value) if value != -1 else 'False'}"
            )
    
    def solve(self):
        if self.hyperoptimizer is None:
            params = self.optimizer.minimize(self.get_expval_func(self.weights), self.angles)
            return self.get_expval_func(self.weights)(params), params

        self.hyperoptimizer.minimize(
            self.get_expval_func, self.optimizer, self.angles, np.array(self.weights))
