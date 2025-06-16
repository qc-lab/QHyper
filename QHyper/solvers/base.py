# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import numpy as np

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult


class SolverConfigException(Exception):
    pass


class SolverException(Exception):
    pass

@dataclass
class SamplesetData:
    """
    Class for storing additional sampleset information.
    Attributes
    ----------
    dwave_sampleset_metadata : np.ndarray
        Record array containing metadata obtained from D-Wave,
        such as:
        - qpu_access_time,
        - qpu_programming_time, etc.

    time_measurements : np.ndarray
        Record array containining information about time measurements of:
        - accessing the clique embedding cache file,
        - creating EmbeddingComposite,
        - .sample() method execution,
    """
    dwave_sampleset_metadata: np.ndarray
    time_measurements: np.ndarray

@dataclass
class SolverResult:
    """
    Class for storing results of the solver.

    Attributes
    ----------
    probabilities : np.recarray
        Record array with the results of the solver. Contains column for each
        variable and the probability of the solution.
    params : dict[Any, Any]
        Dictionary with the best parameters for the solver.
    history : list[list[float]]
        History of the solver. Each element of the list represents the values
        of the objective function at each iteration - there can be multiple
        results per each iteration (epoch).
    sampleset_info : Optional[SamplesetInfo]
        Additional information about the sampleset in case of sampling-based
        methods such as with quantum annealing.
    """
    probabilities: np.recarray
    params: dict[Any, Any]
    history: list[list[OptimizationResult]] = field(default_factory=list)
    sampleset_info: Optional[SamplesetData] = None


class Solver(ABC):
    """
    Abstract base class for solvers.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    """

    problem: Problem

    def __init__(self, problem: Problem):
        self.problem = problem

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'Solver':
        return cls(problem, **config)

    @abstractmethod
    def solve(
            self,
    ) -> SolverResult:
        """
        Parameters are specified in solver implementation.

        Returns
        -------
        SolverResult
            Result of the solver.
        """

        ...
