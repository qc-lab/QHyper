from abc import abstractmethod
from dataclasses import dataclass

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import Optimizer


class SolverConfigException(Exception):
    pass


@dataclass
class SolverResult:
    """
    Class for storing results of the solver.

    Attributes
    ----------
    results_probabilities : dict[str, float]
        Dictionary with results and their probabilities.
    params : dict[Any, Any]
        Dictionary with the best parameters for the solver.
    """
    results_probabilities: dict[str, float]
    params: dict[Any, Any]


class Solver:
    """
    Abstract base class for solvers.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    hyper_optimizer: Optimizer, optional
        Hyperparameter optmizer.
    params_init: dict[str, Any], optional
        Initial parameters for the optimization.
    """

    problem: Problem
    hyper_optimizer: Optional[Optimizer] = None
    params_inits: Optional[dict[str, Any]] = None

    def __init__(self, problem: Problem, **kwargs: Any) -> None:
        pass

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'Solver':
        return cls(problem, **config)

    @abstractmethod
    def solve(
            self,
            params_inits: Optional[dict[str, Any]] = None
    ) -> SolverResult:
        """
        Parameters
        ----------
        params_inits : dict[str, Any], optional
            Initial parameters for the optimization

        Returns
        -------
        SolverResult
            Result of the solver.
        """

        ...
