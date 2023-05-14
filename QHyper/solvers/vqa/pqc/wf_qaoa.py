from dataclasses import dataclass
import pennylane as qml
import numpy as np

import numpy.typing as npt
from typing import Callable, cast

from QHyper.problems.base import Problem

from QHyper.solvers.vqa.pqc.qaoa import QAOA
from QHyper.solvers.converter import Converter
from QHyper.solvers.vqa.eval_funcs.wfeval import WFEval


@dataclass
class WFQAOA(QAOA):
    layers: int = 3
    penalty: float = 0
    mixer: str = "pl_x_mixer"
    backend: str = "default.qubit"

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
        # const_params = params_config['weights']
        probs = self.get_probs_func(problem, list(hyper_args))(
            opt_args.reshape(2, -1))
        results_by_probabilites = {
            format(result, 'b').zfill(len(problem.variables)): float(prob)
            for result, prob in enumerate(probs)
        }
        return WFEval(self.penalty).evaluate(
            results_by_probabilites, problem, list(hyper_args))
