from dataclasses import dataclass, field
import pennylane as qml
from pennylane import numpy as np

import numpy.typing as npt
from typing import Any, Callable, cast, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult

from QHyper.solvers.vqa.pqc.base import PQC
from QHyper.solvers.converter import QUBO, Converter

from .mixers import MIXERS_BY_NAME


@dataclass
class QAOA(PQC):
    layers: int = 3
    backend: str = "default.qubit"
    mixer: str = 'pl_x_mixer'
    qubo_cache: dict[tuple[float], qml.Hamiltonian] = field(
        default_factory=dict)
    dev: qml.Device | None = None

    def create_qubo(self, problem: Problem, weights: list[float]) -> QUBO:
        if tuple(weights) not in self.qubo_cache:
            qubo = Converter.create_qubo(problem, weights)
            self.qubo_cache[tuple(weights)] = self._create_cost_operator(qubo)
        return self.qubo_cache[tuple(weights)]

    def _create_cost_operator(self, qubo: QUBO) -> qml.Hamiltonian:
        result = None
        const = 0

        for variables, coeff in qubo.items():
            if not variables:
                const += coeff
                continue

            summand = None
            for var in variables:
                if summand and str(var) in summand.wires:
                    continue
                encoded_var = (
                    0.5 * qml.Identity(str(var))
                    - 0.5 * qml.PauliZ(str(var))
                )
                summand = (
                    summand @ encoded_var if summand else coeff * encoded_var
                )
            result = result + summand if result else summand
        return result + const * qml.Identity(result.wires[0])

    def _hadamard_layer(self, problem: Problem) -> None:
        for i in problem.variables:
            qml.Hadamard(str(i))

    def _create_mixing_hamiltonian(self, problem: Problem) -> qml.Hamiltonian:
        if self.mixer not in MIXERS_BY_NAME:
            raise Exception(f"Unknown {self.mixer} mixer")
        return MIXERS_BY_NAME[self.mixer]([str(v) for v in problem.variables])

    def _circuit(self, problem: Problem, params: npt.NDArray[np.float64],
                 cost_operator: qml.Hamiltonian) -> None:

        def qaoa_layer(gamma: list[float], beta: list[float]) -> None:
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(
                beta, self._create_mixing_hamiltonian(problem))

        self._hadamard_layer(problem)
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def get_expval_circuit(self, problem: Problem, weights: list[float]
                           ) -> Callable[[npt.NDArray[np.float64]], float]:
        cost_operator = self.create_qubo(problem, weights)

        @qml.qnode(self.dev)
        def expval_circuit(params: npt.NDArray[np.float64]) -> float:
            self._circuit(problem, params, cost_operator)
            return cast(float, qml.expval(cost_operator))

        return cast(Callable[[npt.NDArray[np.float64]], float], expval_circuit)

    def get_probs_func(self, problem: Problem, weights: list[float]
                       ) -> Callable[[npt.NDArray[np.float64]], list[float]]:
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
        def probability_circuit(params: npt.NDArray[np.float64]
                                ) -> list[float]:
            self._circuit(problem, params, cost_operator)
            return cast(list[float],
                        qml.probs(wires=[str(x) for x in problem.variables]))

        return cast(Callable[[npt.NDArray[np.float64]], list[float]],
                    probability_circuit)

    def run_opt(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> float:
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables])
        results = self.get_expval_circuit(problem, list(hyper_args))(
            opt_args.reshape(2, -1))
        return OptimizationResult(results, opt_args)

    def run_with_probs(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables])
        probs = self.get_probs_func(problem, list(hyper_args))(
            opt_args.reshape(2, -1))
        return {
            format(result, 'b').zfill(len(problem.variables)): float(prob)
            for result, prob in enumerate(probs)
        }

    def get_opt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        return args if args is not None else np.array(params_init['angles'])

    def get_hopt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        return (
            hyper_args if hyper_args is not None
            else np.array(params_init['hyper_args'])
        )

    def get_params_init_format(
        self,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        return {
            'angles': opt_args,
            'hyper_args': hyper_args,
        }
