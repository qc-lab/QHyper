from dataclasses import dataclass
import pennylane as qml
import numpy as np

from typing import Any

from QHyper.problems.base import Problem
from .parser import parse_hamiltonian

from QHyper.solvers.vqa.pqc.base import PQC, PQCResults


@dataclass
class QAOA(PQC):
    layers: int = 3
    mixer: str = "X"
    backend: str = "default.qubit"
 
    def _create_cost_operator(self, problem: Problem, weights: list[float]) -> qml.Hamiltonian:
        elements = [problem.objective_function] + problem.constraints
        cost_operator = ""
        if len(weights) != len(elements):
            raise Exception(
                f"Number of provided weights ({len(weights)}) is different from number of elements ({len(elements)})")
        for weight, ingredient in zip(weights, elements):
            cost_operator += f"+{weight}*({ingredient})"
        return parse_hamiltonian(cost_operator)

    def _hadamard_layer(self, problem: Problem):
        for i in range(problem.variables):
            qml.Hadamard(i)

    def _create_mixing_hamiltonian(self, problem: Problem) -> qml.Hamiltonian:
        if self.mixer == "X":
            return qml.qaoa.x_mixer(range(problem.variables))
        # REQUIRES GRAPH https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.xy_mixer.html
        # if self.mixer == "XY": 
        #     return qml.qaoa.xy_mixer(...)
        raise Exception(f"Unknown {self.mixer} mixer")

    def _circuit(self, problem: Problem, params: tuple[list[float], list[float]], cost_operator: qml.Hamiltonian):
        def qaoa_layer(gamma, beta):
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(beta, self._create_mixing_hamiltonian(problem))

        self._hadamard_layer(problem)
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def get_probs_func(self, problem: Problem, weights):
        """Returns function that takes angles and returns probabilities 

        Parameters
        ----------
        weights : list[float]
            weights for converting Problem to QUBO
        
        Returns
        -------
        Callable[[list[float]], float]
            Returns function that takes angles and returns probabilities
        """

        cost_operator = self._create_cost_operator(problem, weights)

        @qml.qnode(self.dev)
        def probability_circuit(params):
            self._circuit(problem, params, cost_operator)
            return qml.probs(wires=range(problem.variables))

        return probability_circuit

    def run(self, problem: Problem, args: list[float], hyper_args: list[float]) -> PQCResults:
        self.dev = qml.device(self.backend, wires=problem.variables)
        # const_params = params_config['weights']
        probs = self.get_probs_func(problem, hyper_args)(np.array(args).reshape(2, -1))

        results_by_probabilites = {
            format(result, 'b').zfill(problem.variables): float(prob) 
            for result, prob in enumerate(probs)
        }
        return results_by_probabilites, hyper_args
    
    def get_params(
            self, 
            params_inits: dict[str, Any], 
            hyper_args: list[float] = []
        ) -> tuple[list[float], list[float]]: 
        return (
            hyper_args if len(hyper_args) > 0 else params_inits['hyper_args'],
            params_inits['angles']
        )
