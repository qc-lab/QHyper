import pennylane as qml
from pennylane import numpy as np

from numpy.typing import NDArray
from typing import Any, Callable, cast

from dataclasses import dataclass, field

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult, Optimizer, Dummy, OptimizationParameter

from QHyper.converter import Converter
from QHyper.polynomial import Polynomial
from QHyper.solvers.base import Solver, SolverResult, SolverException
from QHyper.solvers.gate_based.pennylane.qaoa import QAOA

from QHyper.util import weighted_avg_evaluation


@dataclass
class H_QAOA(QAOA):
    """
    Clasic QAOA implementation.

    Attributes
    ----------
    layers : int
        Number of layers.
    backend : str
        Backend for PennyLane.
    mixer : str
        Mixer name.
    qubo_cache : dict[tuple[float, ...], qml.Hamiltonian]
        Cache for QUBO.
    dev : qml.Device
        PennyLane device instance.
    """

    problem: Problem
    optimizer: Optimizer
    gamma: OptimizationParameter
    beta: OptimizationParameter
    weights: OptimizationParameter
    limit_results: int | None = None
    penalty: float = 0
    layers: int = 3
    backend: str = "default.qubit"
    mixer: str = "pl_x_mixer"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.Device | None = field(default=None, init=False)

    def __init__(
            self, problem: Problem,
            gamma: OptimizationParameter,
            beta: OptimizationParameter,
            weights: OptimizationParameter,
            penalty: float,
            layers: int = 3,
            backend: str = "default.qubit",
            mixer: str = "pl_x_mixer",
            limit_results: int | None = None,
            optimizer: Optimizer = Dummy(),
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.weights = weights
        self.penalty = penalty
        self.limit_results = limit_results
        self.layers = layers
        self.backend = backend
        self.mixer = mixer
        self.qubo_cache = {}

    def get_expval_circuit(
            self) -> Callable[[list[float], list[float]], float]:
        def wrapper(params: list[float]) -> float:
            angles = params[:2*self.layers]
            weights = params[2*self.layers:]

            weights_ = []

            for weight in weights:
                if isinstance(weight, np.numpy_boxes.ArrayBox):
                    weights_.append(weight._value)
                else:
                    weights_.append(weight)
            weights = weights_

            print(angles, weights)

            cost_operator = self.create_cost_operator(self.problem, weights)
            self.dev = qml.device(self.backend, wires=cost_operator.wires)
            probs_func = self.get_probs_func(self.problem, weights)

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
    # def get_opt_args(
    #     self,
    #     params_init: dict[str, Any],
    #     args: Optional[NDArray] = None,
    #     hyper_args: Optional[NDArray] = None,
    # ) -> NDArray:
    #     return np.array(
    #         args if args is not None
    #         else np.array(params_init["angles"])
    #     ).flatten()
    #
    # def get_hopt_args(
    #     self,
    #     params_init: dict[str, Any],
    #     args: Optional[NDArray] = None,
    #     hyper_args: Optional[NDArray] = None,
    # ) -> NDArray:
    #     return (
    #         hyper_args
    #         if hyper_args is not None
    #         else np.array(params_init["hyper_args"])
    #     )

    # def get_params_init_format(
    #     self, opt_args: NDArray, hyper_args: NDArray
    # ) -> dict[str, Any]:
    #     return {
    #         "angles": opt_args,
    #         "hyper_args": hyper_args,
    #     }

    def solve(self, weights: list[float] | None = None,
              gamma: list[float] | None = None,
              beta: list[float] | None = None) -> SolverResult:
        # if gamma is None and self.gamma is None:
        #     raise SolverException("Parameter 'gamma' was not provided")
        # if beta is None and self.beta is None:
        #     raise SolverException("Parameter 'beta' was not provided")

        # gamma = self.gamma if gamma is None else gamma
        # beta = self.beta if beta is None else beta
        if weights is not None:
            weights = self.weights.update(init=weights)
        else:
            weights = self.weights
        gamma_ = self.gamma if gamma is None else self.gamma.update(init=gamma)
        beta_ = self.beta if beta is None else self.beta.update(init=beta)

        # opt_wrapper = LocalOptimizerFunction(
        #         self.pqc, self.problem, best_hargs)
        # opt_res = self.optimizer.minimize(opt_wrapper, opt_args)
        # func = self.get_expval_circuit(weights)

        # opt_res = self.optimizer.minimize(func, angles)
        # assert gamma
        # assert beta
        params = gamma_ + beta_ + self.weights

        opt_res = self.optimizer.minimize(
            self.get_expval_circuit(), params)

        angles = opt_res.params[:2*self.layers]
        weights_res = opt_res.params[2*self.layers:]

        gamma_res = angles[:len(angles) // 2]
        beta_res = angles[len(angles) // 2:]

        return SolverResult(
            self.run_with_probs(self.problem, angles, weights_res),
            {'gamma': gamma_res, 'beta': beta_res, 'weights': weights_res},
            opt_res.history,
        )
