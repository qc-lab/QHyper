from abc import ABC, abstractmethod
from typing import Callable, Any

from ..QAOA_problems.problem import Problem

class Optimizer(ABC):
    func: Callable

    @abstractmethod
    def set_func_from_problem(self, problem: Problem, hyperparameters: dict[str, Any]):
        pass

    def set_func_from_optimizer(self, optimizer: 'Optimizer', init_args: list[float]):
        raise Exception(f"Unable to optimize Optimizer")

    def handle_problem(self, problem: Problem, hyperparameters: dict[str, Any]):
        raise Exception(f"Unable to handle Problem class")

    @abstractmethod
    def minimize(self, init):
        pass
