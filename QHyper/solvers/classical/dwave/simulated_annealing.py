# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult
from QHyper.converter import Converter
from QHyper.constraint import Polynomial

from dwave.samplers import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel


@dataclass
class SimulatedAnnealing(Solver):
    """
    Class for solving a problem using Simulated Annealing (classical solver).

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    penalty_weights : list[float] | None, default None
        The penalty weights for constraints.
    num_reads : int, default 1
        The number of times the solver is run.
    num_sweeps : int, default 1000
        The number of sweeps or steps in the simulated annealing algorithm.
    beta_range : tuple[float, float] | None, default None
        A 2-tuple defining the beginning and end of the beta schedule.
        Beta is the inverse temperature. If not specified, set based on
        the total bias associated with each node.
    """

    problem: Problem
    penalty_weights: list[float] | None = None
    num_reads: int = 1
    num_sweeps: int = 1000
    beta_range: tuple[float, float] | None = None

    def __init__(self,
                 problem: Problem,
                 penalty_weights: list[float] | None = None,
                 num_reads: int = 1,
                 num_sweeps: int = 1000,
                 beta_range: tuple[float, float] | None = None,
                 **config: Any) -> None:
        self.problem = problem
        self.penalty_weights = penalty_weights
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.config = config
        self.sampler = SimulatedAnnealingSampler()

    def solve(self, penalty_weights: list[float] | None = None) -> SolverResult:
        """
        Solve the problem using Simulated Annealing.

        Parameters
        ----------
        penalty_weights : list[float] | None, optional
            Override the penalty weights for this solve call.

        Returns
        -------
        SolverResult
            The result of the solver.
        """
        if penalty_weights is None and self.penalty_weights is None:
            penalty_weights = [1.] * (len(self.problem.constraints) + 1)
        penalty_weights = self.penalty_weights if penalty_weights is None else penalty_weights

        qubo = Converter.create_qubo(self.problem, penalty_weights)
        qubo_terms, offset = convert_qubo_keys(qubo)
        bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

        sample_kwargs = {
            'num_reads': self.num_reads,
            'num_sweeps': self.num_sweeps,
        }

        if self.beta_range is not None:
            sample_kwargs['beta_range'] = self.beta_range

        sample_kwargs.update(self.config)

        sampleset = self.sampler.sample(bqm, **sample_kwargs)

        result = np.recarray(
            (len(sampleset),),
            dtype=([(v, int) for v in sampleset.variables]
                   + [('probability', float)]
                   + [('energy', float)])
        )

        num_of_shots = sampleset.record.num_occurrences.sum()
        for i, solution in enumerate(sampleset.data()):
            for var in sampleset.variables:
                result[var][i] = solution.sample[var]

            result['probability'][i] = (
                solution.num_occurrences / num_of_shots)
            result['energy'][i] = solution.energy

        return SolverResult(result, {"penalty_weights": penalty_weights}, [])


def convert_qubo_keys(qubo: Polynomial) -> tuple[dict[tuple, float], float]:
    new_qubo = defaultdict(float)
    offset = 0.0

    qubo, offset = qubo.separate_const()
    for k, v in qubo.terms.items():
        if len(k) == 1:
            new_key = (k[0], k[0])
        elif len(k) > 2:
            raise ValueError("Only supports quadratic model")
        else:
            new_key = k

        new_qubo[new_key] += v

    return (new_qubo, offset)
