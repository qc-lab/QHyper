import os
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler

from typing import Any, Optional

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer



token = os.environ['DWAVE_API_TOKEN']


class CQM(Solver):
    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(
            self,
            params_inits: dict[str, Any] = None,
            hyper_optimizer: Optional[Optimizer] = None
    ) -> Any:
        converter = Converter()
        cqm = converter.to_cqm(self.problem)
        sampler = LeapHybridCQMSampler(token=token)
        solutions = sampler.sample_cqm(cqm, self.time)
        correct_solutions = [
            s for s in solutions
            if len(cqm.violations(s, skip_satisfied=True)) == 0
        ]

        return correct_solutions[0]
