from dwave.system import LeapHybridDQMSampler

from typing import Any, Optional

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer

import os


token = os.environ["DWAVE_API_TOKEN"]


class DQM(Solver):
    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(
        self,
        params_inits: dict[str, Any] = None,
        hyper_optimizer: Optional[Optimizer] = None,
    ) -> Any:
        converter = Converter()
        sampler = LeapHybridDQMSampler(token=token)

        dqm = converter.to_dqm(self.problem)
        sampleset = sampler.sample_dqm(dqm, self.time)

        return sampleset

    def solve_from_graph(
        self,
        params_inits: dict[str, Any] = None,
        hyper_optimizer: Optional[Optimizer] = None,
    ) -> Any:
        converter = Converter()
        sampler = LeapHybridDQMSampler(token=token)

        dqm = converter.from_graph_to_dqm(self.problem)
        sampleset = sampler.sample_dqm(dqm, self.time)

        return sampleset
