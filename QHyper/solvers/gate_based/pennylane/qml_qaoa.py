# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field

from numpy.typing import NDArray
from typing import Any, cast, Callable

import pennylane as qml

from QHyper.problems.base import Problem
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.optimizers import (
    OptimizationResult, Optimizer, OptimizationParameter)
from QHyper.optimizers.qml_gradient_descent import (
    QML_GRADIENT_DESCENT_OPTIMIZERS)

from QHyper.solvers.gate_based.pennylane.qaoa import QAOA


@dataclass
class QML_QAOA(QAOA):
    """
    QAOA implementation with additonal support for PennyLane optimizers.

    Attributes
    ----------
    layers : int, default 3
        Number of layers.
    mixer : str, default "pl_x_mixer"
        Mixer name.
    backend : str, default "default.qubit"
        Backend device for PennyLane.
    optimizer : str, default ""
        Optimizer name.
    optimizer_args : dict[str, Any], default {}
        Optimizer arguments.
    """
    problem: Problem
    optimizer: Optimizer
    gamma: OptimizationParameter
    beta: OptimizationParameter
    weights: NDArray | None = None
    layers: int = 3
    mixer: str = "pl_x_mixer"
    backend: str = "default.qubit"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.Device | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        print(self.optimizer)
        if not isinstance(self.optimizer, QmlGradientDescent):
            raise ValueError(f"Optimizer {self.optimizer} not supported")

    def get_expval_circuit(
        self, weights: list[float]
    ) -> Callable[[list[float]], OptimizationResult]:
        cost_operator = self.create_cost_operator(self.problem, weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

        @qml.qnode(self.dev)
        def expval_circuit(angles: list[float]) -> Any:
            self._circuit(angles, cost_operator)
            return qml.expval(cost_operator)

        return expval_circuit

    def _run_optimizer(
            self,
            weights: list[float],
            angles: OptimizationParameter
    ) -> OptimizationResult:
        return self.optimizer.minimize_expval_func(
            cast(qml.QNode, self.get_expval_circuit(weights)), angles)
