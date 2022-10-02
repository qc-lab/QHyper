from typing import Callable, Any

from pennylane import numpy as np
import pennylane as qml

from .optimizer import Optimizer
from ..QAOA_problems.problem import Problem


class QmlGradientDescent(Optimizer):
    def __init__(self, optimization_steps: int, optimizer: qml.GradientDescentOptimizer) -> None:
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer

    def minimize(self, init: list[float]) -> list[float]:
        params = np.array(init, requires_grad=True)
        for _ in range(self.optimization_steps):
            params = self.optimizer.step(self.func, params)
        
        return params
    
    def set_func_from_problem(self, problem: Problem, hyperparameters: dict[str, Any]):
        self.func = problem.get_expval_func(**hyperparameters)
    
    # def get_function(self, qaoa_problem, init: list[float]):
    #     def wrapper(hyperparameters: list[float], layers: int):
    #         result = self.minimize(qaoa_problem.get_expval_func(hyperparameters), init)
            

