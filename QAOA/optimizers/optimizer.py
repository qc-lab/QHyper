from abc import ABC, abstractmethod

from ..problem_solver import ProblemSolver

class HyperparametersOptimizer(ABC):
    solver: ProblemSolver

    def __init__(self, solver) -> None:
        self.solver = solver
        epochs: int = 10
    
    @abstractmethod
    def minimize(self):
        pass
