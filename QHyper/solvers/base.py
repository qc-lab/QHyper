from abc import abstractmethod

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import OPTIMIZERS_BY_NAME
from QHyper.optimizers.base import Optimizer


class Solver:
    """
    Abstract base class for solvers.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    hyper_optimizer: Optimizer, optional
        Hyperparameter optmizer.
    """

    problem: Problem
    hyper_optimizer: Optimizer = None

    def __init__(self, problem: Problem, **kwargs: Any) -> None:
        pass

    @staticmethod
    def from_config(problem: Problem, config: dict[str, Any]) -> 'Solver':
        """
        Alternative way of creating solver.
        Expect dict with two keys:
        - type - type of solver
        - args - arguments which will be passed to Solver instance

        Parameters
        ----------
        problem : Problem
            The problem to be solved
        config : dict[str. Any]
            Configuration in form of dict

        Returns
        -------
        Solver
            Initialized Solver object
        """
        from QHyper.solvers import SOLVERS

        try:
            solver_type = config['solver'].pop('type').lower()
            solver_args = config['solver'].pop('args')
        except KeyError:
            raise Exception("Configuration for Solver was not provided")

        solver = SOLVERS[solver_type](problem, **solver_args)

        if 'hyper_optimizer' in config:
            try:
                optimizer_type = config['hyper_optimizer'].pop('type')
            except KeyError:
                raise Exception("Type for hyper optimizer was not provided")

            solver.hyper_optimizer = OPTIMIZERS_BY_NAME[optimizer_type](
                **config['hyper_optimizer']
            )
        return solver

    @abstractmethod
    def solve(
            self,
            params_inits: dict[str, Any]
    ) -> Any:
        """
        Parameters
        ----------
        params_inits : dict[str, Any], optional
            Initial parameters for the optimization. Default is None.

        Returns
        -------
        Any
            The result of the solver.
        """

        ...
