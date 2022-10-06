from abc import ABC, abstractmethod
from typing import Callable, Any

from ..QAOA_problems.problem import Problem

class Optimizer(ABC):
    @abstractmethod
    def minimize(self, func, init):
        pass


class HyperparametersOptimizer(ABC):
    # problem: Problem
    # optimizer: Optimizer

    def minimize(self, func_creator, optimizer, init, hyperparams_init):
        pass
