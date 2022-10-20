from typing import Callable, Any

from pennylane import numpy as np
import pennylane as qml

from .optimizer import Optimizer, ArgsType


class QmlGradientDescent(Optimizer):
    """Gradient Descent Optimizer
    Using GradientDescentOptimizer from library pennylane. 

    Args:
        - optimization_steps - an integer indicating amount of optimization steps
        - optimizer - object of class GradientDescentOptimizer or inheriting from this class 
    """
    def __init__(self, optimization_steps: int, optimizer: qml.GradientDescentOptimizer) -> None:
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer

    def minimize(self, func: Callable[[ArgsType], float], init: ArgsType) -> ArgsType:
        """This method receives:
            - func - function, which will be minimize
            - init - initial args for optimizer

        Returns params which leads to the lowest value of the provided function
        """
        params = np.array(init, requires_grad=True)
        if "reset" in dir(self.optimizer):
            self.optimizer.reset()
        for _ in range(self.optimization_steps):
            params = self.optimizer.step(func, params)
        
        return params
