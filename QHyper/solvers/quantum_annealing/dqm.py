# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import os
import numpy as np
from typing import Any

from dwave.system import LeapHybridDQMSampler
from QHyper.problems import Problem
from QHyper.solvers import Solver, SolverResult
from QHyper.converter import Converter


DWAVE_API_TOKEN = os.environ.get('DWAVE_API_TOKEN')


class DQM(Solver):
    """
    DQM solver class.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    time : float
        Maximum run time in seconds
    cases: int, default 1
        Number of variable cases (values)
        1 is denoting binary variable.
    """

    def __init__(self, problem: Problem, time: float, cases: int = 1) -> None:
        self.problem: Problem = problem
        self.time: float = time
        self.cases: int = cases

    def solve(self, params_inits: dict[str, Any] = {}) -> SolverResult:
        dqm = Converter.to_dqm(self.problem, self.cases)
        sampler = LeapHybridDQMSampler(token=DWAVE_API_TOKEN)
        solutions = sampler.sample_dqm(dqm, self.time)

        recarray = np.recarray(
            (len(solutions),),
            dtype=([(v, int) for v in solutions.variables]
                   + [('probability', float)]
                   + [('energy', float)])
        )

        num_of_shots = solutions.record.num_occurrences.sum()
        for i, solution in enumerate(solutions.data()):
            for var in solutions.variables:
                recarray[var][i] = solution.sample[var]

            recarray['probability'][i] = (
                solution.num_occurrences / num_of_shots)
            recarray['energy'][i] = solution.energy

        return SolverResult(
            recarray,
            {},
            []
        )
