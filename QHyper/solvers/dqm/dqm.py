from dwave.system import LeapHybridDQMSampler

from typing import Any

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver

import os


token = os.environ["DWAVE_API_TOKEN"]


class DQM(Solver):
    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        converter = Converter()
        dqm = converter.to_dqm(self.problem)
        sampler = LeapHybridDQMSampler(token=token)
        sampleset = sampler.sample_dqm(dqm, self.time)

        return sampleset
