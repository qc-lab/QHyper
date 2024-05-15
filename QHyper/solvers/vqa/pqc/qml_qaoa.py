# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field

from numpy.typing import NDArray
from typing import Any, cast

from pennylane import QNode

from QHyper.problems.base import Problem
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.optimizers.base import OptimizationResult

from QHyper.solvers.vqa.pqc.qaoa import QAOA


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

    layers: int = 3
    mixer: str = "pl_x_mixer"
    backend: str = "default.qubit"
    optimizer: str = ''
    optimizer_args: dict[str, Any] = field(default_factory=dict)

    def run_opt(
            self,
            problem: Problem,
            opt_args: NDArray,
            hyper_args: NDArray
    ) -> OptimizationResult:
        if self.optimizer == '':
            raise ValueError("Optimizer not provided, if you don't "
                             "want to use optimizer use qaoa instead")
        optimizer_instance = QmlGradientDescent(
            self.optimizer, **self.optimizer_args)

        return optimizer_instance.minimize_expval_func(
            cast(QNode, self.get_expval_circuit(problem, hyper_args)),
            opt_args
        )
