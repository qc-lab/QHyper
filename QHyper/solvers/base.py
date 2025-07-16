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
        Record array containing metadata obtained from D-Wave:
        - qpu_sampling_time_us,
        - qpu_anneal_time_per_sample_us,
        - qpu_readout_time_per_sample_us,
        - qpu_access_time_us,
        - qpu_access_overhead_time_us,
        - qpu_programming_time_us,
        - qpu_delay_time_per_sample_us,
        - total_post_processing_time_us,
        - post_processing_overhead_time_us,

    The time units are microseconds (us) according to the D-Wave Docs (July 2025):
    https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html. 
    

    time_measurements : np.ndarray
        Record array containining information about time measurements of:
        - find_clique_embedding_time_s - in case of clique embedding:
        first call to that function after installment results
        in the embedding search (might take minutes), next calls are accessing the clique
        emedding cache file,
        or
        - find_heuristic_embedding_time_s - in case of heuristic embedding: search for heuristic embedding,
        - fixed_embedding_composite_time_s - creating FixedEmbeddingComposite object,
        - sample_func_time_s - method execution of the .sample function - communication with the solver itself.
        (https://dwave-systemdocs.readthedocs.io/en/link_fix/reference/composites/generated/dwave.system.composites.FixedEmbeddingComposite.sample.html#dwave.system.composites.FixedEmbeddingComposite.sample),

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
    sampleset_info : Optional[SamplesetData]
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
