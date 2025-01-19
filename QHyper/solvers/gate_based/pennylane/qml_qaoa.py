# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field

from typing import Any, cast, Callable

import pennylane as qml

from QHyper.problems.base import Problem
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.optimizers import (
    OptimizationResult, Optimizer, OptimizationParameter)

from QHyper.solvers.gate_based.pennylane.qaoa import QAOA


@dataclass
class QML_QAOA(QAOA):
    """
    QAOA implementation with additonal support for PennyLane optimizers.

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
    penalty_weights : list[float] | None
        Penalty weights used for converting Problem to QUBO. They connect cost function
        with constraints. If not specified, all penalty weights are set to 1.
    backend : str
        Backend for PennyLane.
    mixer : str
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
    optimizer: Optimizer
    penalty_weights: list[float] | None = None
    mixer: str = "pl_x_mixer"
    backend: str = "default.qubit"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.devices.LegacyDevice | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, QmlGradientDescent):
            raise ValueError(f"Optimizer {self.optimizer} not supported")

    def get_expval_circuit(
        self, penalty_weights: list[float]
    ) -> Callable[[list[float]], OptimizationResult]:
        cost_operator = self.create_cost_operator(
            self.problem, penalty_weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

        @qml.qnode(self.dev)
        def expval_circuit(angles: list[float]) -> Any:
            self._circuit(angles, cost_operator)
            return qml.expval(cost_operator)

        return expval_circuit

    def _run_optimizer(
            self,
            penalty_weights: list[float],
            angles: OptimizationParameter
    ) -> OptimizationResult:
        return self.optimizer.minimize_expval_func(
            cast(qml.QNode, self.get_expval_circuit(penalty_weights)), angles)
