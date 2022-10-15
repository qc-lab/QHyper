from typing import Callable, Any

from pennylane import numpy as np
import pennylane as qml

from .optimizer import Optimizer
from ..QAOA_problems.problem import Problem


class QmlGradientDescent(Optimizer):
    def __init__(self, optimization_steps: int, optimizer: qml.GradientDescentOptimizer) -> None:
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer

    def minimize(self, func: Callable, init: list[float]) -> list[float]:
        params = np.array(init, requires_grad=True)
        if "reset" in dir(self.optimizer):
            self.optimizer.reset()
        for _ in range(self.optimization_steps):
            params = self.optimizer.step(func, params)
        
        return params
    
    # def get_function(self, qaoa_problem, init: list[float]):
    #     def wrapper(hyperparameters: list[float], layers: int):
    #         result = self.minimize(qaoa_problem.get_expval_func(hyperparameters), init)
            

