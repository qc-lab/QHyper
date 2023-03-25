from dataclasses import dataclass
import pennylane as qml
import numpy as np

from typing import Any

from QHyper.problems.base import Problem

from QHyper.solvers.vqa.pqc.base import PQC, PQCResults
from QHyper.solvers.converter import QUBO, Converter


@dataclass
class HQAOA(PQC):
    layers: int = 3
    mixer: str = "X"
    backend: str = "default.qubit"
    
    def _create_cost_operator(self, qubo: QUBO) -> qml.Hamiltonian:
        result = qml.Identity(0) 
        for variables, coeff in qubo.items():
            if not variables:
                continue
            tmp = coeff * (0.5 * qml.Identity(str(variables[0])) - 0.5 * qml.PauliZ(str(variables[0])))
            if len(variables) == 2 and variables[0] != variables[1]:
                tmp = tmp @ (0.5 * qml.Identity(str(variables[1])) - 0.5 * qml.PauliZ(str(variables[1])))
            result += tmp
        return result

    def _hadamard_layer(self, problem: Problem):
        for i in problem.variables:
            qml.Hadamard(str(i))

    def _create_mixing_hamiltonian(self, problem: Problem) -> qml.Hamiltonian:
        if self.mixer == "X":
            return qml.qaoa.x_mixer([str(x) for x in problem.variables])
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
        qubo = Converter.create_qubo(problem, weights)
        cost_operator = self._create_cost_operator(qubo)

        @qml.qnode(self.dev)
        def probability_circuit(params):
            self._circuit(problem, params, cost_operator)
            return qml.probs(wires=[str(x) for x in problem.variables])

        return probability_circuit

    def run(self, problem: Problem, args: list[float], hyper_args: list[float]) -> PQCResults:
        self.dev = qml.device(self.backend, wires=[str(x) for x in problem.variables])
        weights = args[:1 + len(problem.constraints)]

        probs = self.get_probs_func(problem, weights)(np.array(args[1 + len(problem.constraints):]).reshape(2, -1))

        results_by_probabilites = {
            format(result, 'b').zfill(len(problem.variables)): float(prob) 
            for result, prob in enumerate(probs)
        }
        return results_by_probabilites, weights

    def get_params(
            self, 
            params_inits: dict[str, Any], 
            hyper_args: list[float] = []
        ) -> tuple[list[float], list[float]]: 
        if len(hyper_args) > 0:
            return (
                hyper_args,
                hyper_args
            )
        else:
            return (
                np.concatenate((params_inits['hyper_args'], np.array(params_inits['angles']).flatten())),
                np.concatenate((params_inits['hyper_args'], np.array(params_inits['angles']).flatten())),
            )
