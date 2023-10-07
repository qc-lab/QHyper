from dataclasses import dataclass
import pennylane as qml
import numpy as np

import numpy.typing as npt
from typing import Any, Callable, cast, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult

from QHyper.solvers.vqa.pqc.qaoa import QAOA
from QHyper.solvers.converter import Converter
from QHyper.util import weighted_avg_evaluation


@dataclass
class HQAOA(QAOA):
    layers: int = 3
    penalty: float = 0
    mixer: str = "pl_x_mixer"
    backend: str = "default.qubit"
    limit_results: int | None = None

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
    ) -> OptimizationResult:
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables])
        weights = list(opt_args[:1 + len(problem.constraints)])
        probs = self.get_probs_func(problem, list(weights))(
            opt_args[1 + len(problem.constraints):].reshape(2, -1))
        results_by_probabilites = {
            format(result, 'b').zfill(len(problem.variables)): float(prob)
            for result, prob in enumerate(probs)
        }
        result = weighted_avg_evaluation(
            results_by_probabilites, problem.get_score, self.penalty,
            limit_results=self.limit_results
        )
        return OptimizationResult(result, opt_args)

    def run_with_probs(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables])
        weights = list(opt_args[:1 + len(problem.constraints)])
        probs = self.get_probs_func(problem, list(weights))(
            opt_args[1 + len(problem.constraints):].reshape(2, -1))
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
        return np.concatenate((
            hyper_args if hyper_args is not None
            else params_init['hyper_args'],
            np.array(args if args else params_init['angles']).flatten()
        ))

    def get_params_init_format(
        self,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        return {
            'angles': opt_args[len(hyper_args):],
            'hyper_args': opt_args[:len(hyper_args)],
        }
