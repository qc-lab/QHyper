from dwave.system import LeapHybridCQMSampler

from ..solver import Solver
from ...problems.problem import Problem


class CQM(Solver):
    def __init__(self, **kwargs) -> None:
        self.problem: Problem = kwargs.get("problem")
        self.time: float = kwargs.get("time", None)

    def solve(self):
        pass
        # cqm = self.problem.to_cqm() #todo
        # sampler = LeapHybridCQMSampler()
        # solutions = sampler.sample_cqm(cqm, self.time)
        # correct_solutions = [s for s in solutions if len(cqm.violations(s, skip_satisfied=True)) == 0]
        #
        # return correct_solutions[0]
