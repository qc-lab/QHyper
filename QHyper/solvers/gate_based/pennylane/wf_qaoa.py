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
from QHyper.util import weighted_avg_evaluation

from QHyper.solvers.gate_based.pennylane.qaoa import QAOA


@dataclass
class WF_QAOA(QAOA):
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
    layers: int
    gamma: OptimizationParameter
    beta: OptimizationParameter
    optimizer: Optimizer
    weights: NDArray | None
    penalty: float = 0
    backend: str = "default.qubit"
    mixer: str = "pl_x_mixer"
    limit_results: int | None = None
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.Device | None = field(default=None, init=False)

    def __init__(
            self,
            problem: Problem,
            layers: int,
            gamma: OptimizationParameter,
            beta: OptimizationParameter,
            weights: NDArray | None = None,
            penalty: float = 0,
            backend: str = "default.qubit",
            mixer: str = "pl_x_mixer",
            limit_results: int | None = None,
            optimizer: Optimizer = Dummy(),
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.penalty = penalty
        self.weights = weights
        self.limit_results = limit_results
        self.layers = layers
        self.backend = backend
        self.mixer = mixer
        self.qubo_cache = {}

    def get_expval_circuit(self, weights: list[float]
                           ) -> Callable[[list[float]], float]:
        cost_operator = self.create_cost_operator(self.problem, weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

        probs_func = self.get_probs_func(self.problem, weights)

        def wrapper(angles: list[float]) -> float:
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
            return OptimizationResult(result, angles)
        return wrapper
