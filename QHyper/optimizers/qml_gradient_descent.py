from typing import Any, Callable

import pennylane as qml
from pennylane import numpy as np

from .optimizer import ArgsType, Optimizer


class QmlGradientDescent(Optimizer):
    """Gradient Descent Optimizer

    Using GradientDescentOptimizer from library PennyLane.

    Attributes
    ----------
    optimization_steps : int
        number of optimization steps
    optimizer : qml.GradientDescentOptimizer
        object of class GradientDescentOptimizer or inheriting from this class
    """

    def __init__(self, optimization_steps: int, optimizer: qml.GradientDescentOptimizer) -> None:
        """
        Parameters
        ----------
        optimization_steps : int
            number of optimization steps
        optimizer : qml.GradientDescentOptimizer
            object of class GradientDescentOptimizer or inheriting from this class
        """

        self.optimization_steps = optimization_steps
        self.optimizer = optimizer

    def minimize(self, func: Callable[[ArgsType], float], init: ArgsType) -> tuple[ArgsType, Any]:
        """Returns params which lead to the lowest value of the provided function and cost history

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
        return params, cost_history
