# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import os
from dwave.system import LeapHybridCQMSampler

from typing import Any

from QHyper.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver


DWAVE_API_TOKEN = os.environ.get('DWAVE_API_TOKEN')


class CQM(Solver):
    name = "CQM1"
    """
    Class for solving a problem using the
    Constrained Quadratic Model (CQM) approach.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    time : float
        The maximum time allowed for the CQM solver.
    """

    def __init__(self, problem: Problem, time: float) -> None:
        """
        Parameters
        ----------
        problem : Problem
            The problem to be solved.
        time : float
            The maximum time allowed for the CQM solver.
        """
        self.problem: Problem = problem
        self.time: float = time

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        """
        Solve the problem using the CQM approach.

        Returns
        -------
        Any
            The solution to the problem.
        """
        converter = Converter()
        cqm = converter.to_cqm(self.problem)
        sampler = LeapHybridCQMSampler(token=DWAVE_API_TOKEN)
        solutions = sampler.sample_cqm(cqm, self.time)
        correct_solutions = [
            s for s in solutions
            if len(cqm.violations(s, skip_satisfied=True)) == 0
        ]

        return correct_solutions[0]
