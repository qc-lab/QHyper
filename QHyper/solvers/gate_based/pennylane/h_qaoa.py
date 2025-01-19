import pennylane as qml
from pennylane import numpy as np

from typing import Callable

from dataclasses import dataclass, field

from QHyper.problems.base import Problem
from QHyper.optimizers import (
        OptimizationResult, Optimizer, Dummy, OptimizationParameter)

from QHyper.solvers.base import SolverResult
from QHyper.solvers.gate_based.pennylane.qaoa import QAOA

from QHyper.util import weighted_avg_evaluation


@dataclass
class H_QAOA(QAOA):
    """
    Different implementation of QAOA.
    This implementation uses different function to evaluate the hamiltonian -
    this function doesn't return expectation value but the score of the
    solution. Another difference is that this implementation update the penalty weights
    of the problem in the optimization process of QAOA.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    layers : int
        Number of layers.
    gamma : OptimizationParameter
        Vector of gamma angles used in cost Hamiltonian. Size of the vector
        should be equal to the number of layers.
    beta : OptimizationParameter
        Vector of beta angles used in mixing Hamiltonian. Size of the vector
        should be equal to the number of layers.
    optimizer : Optimizer
        Optimizer used in the classical part of the algorithm.
    penalty_weights : OptimizationParameter
        Penalty Weights used for converting Problem to QUBO. They connect cost function
        with constraints. If not specified, all penalty weights are set to 1. But
        unlike in QAOA, this parameter is updated during optimization.
    limit_results : int | None, default None
        Specifies how many results should be considered in the evaluation of
        the objective function. If None, all results are considered.
    penalty : float, default 0
        When calculating the score of the solution, the penalty is the score
        for the solution that doesn't satisfy the constraints.
    backend : str, default 'default.qubit'
        Backend for PennyLane.
    mixer : str, default 'pl_x_mixer'
        Mixer name. Currently only 'pl_x_mixer' is supported.
    qubo_cache : dict[tuple[float, ...], qml.Hamiltonian]
        Cache for QUBO.
    dev : qml.devices.LegacyDevice
        PennyLane device instance.
    """

    problem: Problem
    layers: int
    gamma: OptimizationParameter
    beta: OptimizationParameter
    penalty_weights: OptimizationParameter
    optimizer: Optimizer
    limit_results: int | None = None
    penalty: float = 0
    backend: str = "default.qubit"
    mixer: str = "pl_x_mixer"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.devices.LegacyDevice | None = field(default=None, init=False)

    def __init__(
            self,
            problem: Problem,
            layers: int,
            gamma: OptimizationParameter,
            beta: OptimizationParameter,
            penalty_weights: OptimizationParameter,
            penalty: float,
            backend: str = "default.qubit",
            mixer: str = "pl_x_mixer",
            limit_results: int | None = None,
            optimizer: Optimizer = Dummy(),
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.penalty_weights = penalty_weights
        self.penalty = penalty
        self.limit_results = limit_results
        self.layers = layers
        self.backend = backend
        self.mixer = mixer
        self.qubo_cache = {}

    def get_expval_circuit(self) -> Callable[[list[float],
                                              list[float]], float]:
        def wrapper(params: list[float]) -> float:
            angles = params[:2*self.layers]
            penalty_weights = params[2*self.layers:]

            penalty_weights_ = []

            for weight in penalty_weights:
                if isinstance(weight, np.numpy_boxes.ArrayBox):
                    penalty_weights_.append(weight._value)
                else:
                    penalty_weights_.append(weight)
            penalty_weights = penalty_weights_

            cost_operator = self.create_cost_operator(self.problem, penalty_weights)
            self.dev = qml.device(self.backend, wires=cost_operator.wires)
            probs_func = self.get_probs_func(self.problem, penalty_weights)

            probs = probs_func(angles)
            if isinstance(probs, np.numpy_boxes.ArrayBox):
                probs = probs._value

            dtype = [
                (wire, 'i4') for wire in self.dev.wires]+[('probability', 'f8')]
            recarray = np.recarray((len(probs),), dtype=dtype)
            for i, probability in enumerate(probs):
                solution = format(i, "b").zfill(self._get_num_of_wires())
                recarray[i] = *solution, probability

            result = weighted_avg_evaluation(
                recarray, self.problem.get_score, self.penalty,
                limit_results=self.limit_results
            )
            return OptimizationResult(result, params)
        return wrapper

    def solve(self, penalty_weights: list[float] | None = None,
              gamma: list[float] | None = None,
              beta: list[float] | None = None) -> SolverResult:
        if penalty_weights is not None:
            penalty_weights = self.penalty_weights.update(init=penalty_weights)
        else:
            penalty_weights = self.penalty_weights
        gamma_ = self.gamma if gamma is None else self.gamma.update(init=gamma)
        beta_ = self.beta if beta is None else self.beta.update(init=beta)

        params = gamma_ + beta_ + self.penalty_weights

        opt_res = self.optimizer.minimize(
            self.get_expval_circuit(), params)

        angles = opt_res.params[:2*self.layers]
        penalty_weights_res = opt_res.params[2*self.layers:]

        gamma_res = angles[:len(angles) // 2]
        beta_res = angles[len(angles) // 2:]

        return SolverResult(
            self.run_with_probs(self.problem, angles, penalty_weights_res),
            {
                'gamma': gamma_res, 'beta': beta_res,
                'penalty_weights': penalty_weights_res
            },
            opt_res.history,
        )
