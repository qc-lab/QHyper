# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import os
import numpy as np
from dwave.system import LeapHybridCQMSampler

from dataclasses import dataclass

from QHyper.converter import Converter
from QHyper.problems import Problem
from QHyper.solvers import Solver, SolverResult


DWAVE_API_TOKEN = os.environ.get('DWAVE_API_TOKEN')


@dataclass
class CQM(Solver):
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

    problem: Problem
    time: float
    token: str | None = None

    def solve(self) -> SolverResult:
        """
        Solve the problem using the CQM approach.

        Returns
        -------
        Any
            The solution to the problem.
        """
        converter = Converter()
        cqm = converter.to_cqm(self.problem)
        sampler = LeapHybridCQMSampler(token=self.token or DWAVE_API_TOKEN)
        solutions = sampler.sample_cqm(cqm, self.time).aggregate()

        recarray = np.recarray(
            (len(solutions),),
            dtype=([(v, int) for v in solutions.variables]
                   + [('probability', float)]
                   + [('energy', float)]
                   + [('is_feasible', bool)])
        )

        num_of_shots = solutions.record.num_occurrences.sum()
        for i, solution in enumerate(solutions.data()):
            for var in solutions.variables:
                recarray[var][i] = solution.sample[var]

            recarray['probability'][i] = (
                solution.num_occurrences / num_of_shots)
            recarray['energy'][i] = solution.energy
            recarray['is_feasible'][i] = solution.is_feasible

        return SolverResult(recarray, {}, [])
