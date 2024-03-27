# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field
import pennylane as qml
import numpy as np

import numpy.typing as npt
from typing import Any

from QHyper.problems.base import Problem
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent

from QHyper.solvers.vqa.pqc.qaoa import QAOA


@dataclass
class QML_QAOA(QAOA):
    layers: int = 3
    mixer: str = "pl_x_mixer"
    backend: str = "default.qubit"
    optimizer: str = ''
    optimizer_args: dict[str, Any] = field(default_factory=dict)

    def run_opt(
            self,
            problem: Problem,
            opt_args: npt.NDArray[np.float64],
            hyper_args: npt.NDArray[np.float64]
    ) -> float:
        if self.optimizer == '':
            raise ValueError("Optimizer not provided, if you don't "
                             "want to use optimizer use qaoa instead")
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables])
        optimizer_instance = QmlGradientDescent(
            self.optimizer, **self.optimizer_args)

        return optimizer_instance.minimize_expval_func(
            self.get_expval_circuit(problem, list(hyper_args)), opt_args)
