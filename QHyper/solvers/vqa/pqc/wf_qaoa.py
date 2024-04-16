# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass
import pennylane as qml
import numpy as np

import numpy.typing as npt
from typing import Callable, cast

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult

from QHyper.solvers.vqa.pqc.qaoa import QAOA

from QHyper.util import weighted_avg_evaluation


@dataclass
class WFQAOA(QAOA):
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
        cost_operator = self.create_qubo(problem, weights)

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
        hyper_args: npt.NDArray[np.float64],
        print_probabilities: bool = False
    ) -> OptimizationResult:
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables])
        probs = self.get_probs_func(problem, list(hyper_args))(
            opt_args.reshape(2, -1))

        if isinstance(probs, qml.numpy.numpy_boxes.ArrayBox):
            probs = probs._value
        results_by_probabilites = {
            format(result, 'b').zfill(len(problem.variables)): float(prob)
            for result, prob in enumerate(probs)
        }
        if print_probabilities:
            sorted_results = {
                k: v for k, v in
                sorted(
                    results_by_probabilites.items(),
                    key=lambda item: item[1],
                    reverse=True
                )[:8]
            }
            for k, v in sorted_results.items():
                print(f'{k}, {v:.3f}, {problem.get_score(k)}')

        result = weighted_avg_evaluation(
            results_by_probabilites, problem.get_score, self.penalty,
            limit_results=self.limit_results
        )
        return OptimizationResult(result, opt_args)
