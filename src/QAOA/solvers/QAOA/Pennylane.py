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
        if len(weights) != len(ingredients):
            raise Exception(
                f"Number of provided weights ({len(weights)}) is different from number of ingredients ({len(ingredients)})")
        for weight, ingredient in zip(weights, ingredients):
            cost_operator += f"+{weight}*({ingredient})"
        return parse_hamiltonian(cost_operator)

    def _hadamard_layer(self):
        for i in range(self.problem.wires):
            qml.Hadamard(i)
    
    def create_mixing_hamitonian(self):
        hamiltonian = qml.Hamiltonian([], [])
        if self.mixer == "X":
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

    def get_probs_val_func(self, weights):
        
        # cost_operator = self.create_cost_operator(weights)
        # @qml.qnode(self.dev)
        def probability_value(params):
            probs = self.get_probs_func(weights)(params)
        #     self.circuit(params, cost_operator)
        #     probs = qml.probs(wires=range(self.problem.wires))
        #     print(probs)
            return self.check_results(probs)

        return probability_value

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
            if (value:=self.problem.get_score(to_bin(result))) is None:
                score += 0 # experiments?
            else:
                score -= prob*value
        return score
    
    def print_results(self, probs):
        to_bin = lambda x: format(x, 'b').zfill(self.problem.wires)
        results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
        results_by_probabilites = dict(
            sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
        for result, prob in results_by_probabilites.items():
            # binary_rep = to_bin(key)
            value = self.problem.get_score(to_bin(result))
            print(
                f"Key: {to_bin(result)} with probability {prob:.5f}   "
                f"| correct: {'True, value: '+format(value, '.5f') if value is not None else 'False'}"
            )
    
    def solve(self):
        weights = self.hyperoptimizer.minimize(
            self.get_expval_func, self.optimizer, self.angles, np.array(self.weights), [0, 100]
            ) if self.hyperoptimizer else self.weights
        
        params = self.optimizer.minimize(self.get_expval_func(weights), self.angles)
        return self.get_expval_func(weights)(params), params, weights
