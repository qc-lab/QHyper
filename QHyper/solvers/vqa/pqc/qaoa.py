# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field
import pennylane as qml
from pennylane import numpy as np

import numpy.typing as npt
from typing import Any, Callable, cast, Optional

from pennylane import wires

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult

from QHyper.solvers.vqa.pqc.base import PQC
from QHyper.converter import Converter
from QHyper.polynomial import Polynomial

from .mixers import MIXERS_BY_NAME


@dataclass
class QAOA(PQC):
    layers: int = 3
    backend: str = "default.qubit"
    mixer: str = "pl_x_mixer"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(default_factory=dict)
    dev: qml.Device | None = None

    def create_cost_operator(self, problem: Problem, weights: list[float]
                             ) -> qml.Hamiltonian:
        if tuple(weights) not in self.qubo_cache:
            qubo = Converter.create_qubo(problem, weights)
            self.qubo_cache[tuple(weights)] = self._create_cost_operator(qubo)
        return self.qubo_cache[tuple(weights)]

    def _create_cost_operator(self, qubo: Polynomial) -> qml.Hamiltonian:
        result: qml.Hamiltonian | None = None
        const = 0
        print('qubo:', type(qubo))

        for variables, coeff in qubo.terms.items():
            if not variables:
                const += coeff
                continue

            summand: qml.Hamiltonian | None = None
            for var in variables:
                if summand and str(var) in summand.wires:
                    continue
                encoded_var = cast(
                    qml.Hamiltonian,
                    0.5 * qml.Identity(str(var)) - 0.5 * qml.PauliZ(str(var))
                )

                summand = summand @ encoded_var if summand else coeff * encoded_var
            result = result + summand if result else summand
        return result + const * qml.Identity(result.wires[0])

    def _hadamard_layer(self, cost_operator: qml.Hamiltonian) -> None:
        for i in cost_operator.wires:
            qml.Hadamard(str(i))

    def _create_mixing_hamiltonian(self, cost_operator: qml.Hamiltonian) -> qml.Hamiltonian:
        if self.mixer not in MIXERS_BY_NAME:
            raise Exception(f"Unknown {self.mixer} mixer")
        return MIXERS_BY_NAME[self.mixer]([str(v) for v in cost_operator.wires])

    def _circuit(
        self,
        params: list[float],
        cost_operator: qml.Hamiltonian,
    ) -> None:
        def qaoa_layer(gamma: list[float], beta: list[float]) -> None:
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(beta, self._create_mixing_hamiltonian(cost_operator))

        self._hadamard_layer(cost_operator)
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def get_expval_circuit(
        self, problem: Problem, weights: list[float]
    ) -> Callable[[list[float]], float]:
        cost_operator = self.create_cost_operator(problem, weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

        @qml.qnode(self.dev)
        def expval_circuit(params: list[float]) -> float:
            self._circuit(params, cost_operator)
            return cast(float, qml.expval(cost_operator))

        return cast(Callable[[list[float]], float], expval_circuit)

    def get_probs_func(
        self, problem: Problem, weights: list[float]
    ) -> Callable[[list[float]], list[float]]:
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
        cost_operator = self.create_cost_operator(problem, weights)

        @qml.qnode(self.dev)
        def probability_circuit(params: list[float]) -> list[float]:
            self._circuit(params, cost_operator)
            return cast(
                list[float], qml.probs(wires=cost_operator.wires)
            )

        return cast(
            Callable[[list[float]], list[float]], probability_circuit
        )

    def run_opt(
        self,
        problem: Problem,
        opt_args: list[float],
        hyper_args: list[float],
    ) -> OptimizationResult:
        # self.dev = qml.device(self.backend, wires=[str(x) for x in problem.variables])
        results = self.get_expval_circuit(problem, hyper_args)(opt_args)
        return OptimizationResult(results, opt_args)

    def run_with_probs(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        self.dev = qml.device(self.backend, wires=[str(x) for x in problem.variables])
        probs = self.get_probs_func(problem, list(hyper_args))(opt_args.reshape(2, -1))
        return {
            format(result, "b").zfill(len(problem.variables)): float(prob)
            for result, prob in enumerate(probs)
        }

    def get_opt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        return args if args is not None else np.array(params_init["angles"])

    def get_hopt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        return (
            hyper_args
            if hyper_args is not None
            else np.array(params_init["hyper_args"])
        )

    def get_params_init_format(
        self, opt_args: npt.NDArray[np.float64], hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        return {
            "angles": opt_args,
            "hyper_args": hyper_args,
        }
