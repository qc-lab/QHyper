from abc import abstractmethod

from dataclasses import dataclass

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers.base import Optimizer


class Solver:
    """
    Abstract base class for solvers.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.

    Methods
    -------
    solve(params_inits=None, hyper_optimizer=None)
        Solve the problem using the solver.
    """

    problem: Problem

    @abstractmethod
    def solve(
            self,
            params_inits: dict[str, Any] = None,
            hyper_optimizer: Optional[Optimizer] = None
    ) -> Any:
        """
        Parameters
        ----------
        params_inits : dict[str, Any], optional
            Initial parameters for the optimization. Default is None.
        hyper_optimizer : Optimizer, optional
            Hyperparameter optimizer. Default is None.

        Returns
        -------
        Any
            The result of the solver.
        """

        ...
