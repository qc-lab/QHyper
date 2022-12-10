from abc import ABC, abstractmethod
from typing import Any


class Solver(ABC):
    """Interface for solvers"""

    @abstractmethod
    def solve(self) -> Any:
        pass
