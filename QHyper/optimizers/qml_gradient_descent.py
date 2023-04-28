from typing import Callable
import numpy.typing as npt

import pennylane as qml
from pennylane import numpy as np

from .base import Optimizer


class QmlGradientDescent(Optimizer):
    """Gradient Descent Optimizer

    Using GradientDescentOptimizer from library PennyLane.

    Attributes
    ----------
    optimizer : qml.GradientDescentOptimizer
        object of class GradientDescentOptimizer or inheriting from this class
    optimization_steps : int
        number of optimization steps
    """

    def __init__(
        self,
        optimizer: qml.GradientDescentOptimizer,
        optimization_steps: int
    ) -> None:
        """
        Parameters
        ----------
        optimizer : qml.GradientDescentOptimizer
            object of class GradientDescentOptimizer
            or inheriting from this class
        optimization_steps : int
            number of optimization steps
        """

        self.optimizer = optimizer
        self.optimization_steps = optimization_steps

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], float],
        init: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """Returns params which lead to the lowest value of
            the provided function and cost history

        Parameters
        ----------
        func : Callable[[ArgsType], float]
            function which will be minimized
        init : ArgsType
            initial args for optimizer

        Returns
        -------
        tuple[ArgsType, Any]
            Returns tuple which contains params taht lead to the lowest value
            of the provided function and cost history
        """

        cost_history = []
        params = np.array(init, requires_grad=True)
        if "reset" in dir(self.optimizer):
            self.optimizer.reset()
        for _ in range(self.optimization_steps):
            params, cost = self.optimizer.step_and_cost(func, params)
            cost_history.append(cost)
        return cost, params
