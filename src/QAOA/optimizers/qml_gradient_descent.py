from typing import Callable, Any

from pennylane import numpy as np
import pennylane as qml

from .optimizer import Optimizer, ArgsType


class QmlGradientDescent(Optimizer):
    """Gradient Descent Optimizer

    Using GradientDescentOptimizer from library pennylane. 

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

    def minimize(self, func: Callable[[ArgsType], float], init: ArgsType) -> ArgsType:
        """Returns params which leads to the lowest value of the provided function 

        Parameters
        ----------
        func : Callable[[ArgsType], float]
            function, which will be minimize
        init : ArgsType
            initial args for optimizer

        Returns
        -------
        params : ArgsType
            Returns args which gave the lowest value.
        """
        
        params = np.array(init, requires_grad=True)
        if "reset" in dir(self.optimizer):
            self.optimizer.reset()
        for _ in range(self.optimization_steps):
            params = self.optimizer.step(func, params)
        
        return params
