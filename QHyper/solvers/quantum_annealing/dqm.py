# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import os
from typing import Any

from dwave.system import LeapHybridDQMSampler
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.converter import Converter

DWAVE_API_TOKEN = os.environ.get("DWAVE_API_TOKEN")


class DQM(Solver):
    """
    DQM solver class.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    time : float
        Maximum run time in seconds.
    """

    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        converter = Converter()
        dqm = converter.to_dqm(self.problem)
        sampler = LeapHybridDQMSampler(token=DWAVE_API_TOKEN)
        sampleset = sampler.sample_dqm(dqm, self.time)

        return sampleset
