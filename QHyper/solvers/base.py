from abc import abstractmethod

from dataclasses import dataclass

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers.base import Optimizer


@dataclass
class SolverResults:
    value: float
    params: list[float]


class Solver:
    problem: Problem
    # config: dict[str, Any]
    """Interface for solvers"""

    @abstractmethod
    def solve(
            self,
            params_inits: dict[str, Any] = None,
            hyper_optimizer: Optional[Optimizer] = None
    ) -> Any:
        ...
