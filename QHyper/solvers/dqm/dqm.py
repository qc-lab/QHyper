from dimod import DiscreteQuadraticModel
from dwave.system import LeapHybridDQMSampler

from typing import Any, Optional

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer



class DQM(Solver):
    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(
            self,
            params_inits: dict[str, Any] = None,
            hyper_optimizer: Optional[Optimizer] = None
    ) -> Any:
        converter = Converter()
        sampler = LeapHybridDQMSampler()
        
        dqm = converter.to_dqm(self.problem)

        sampleset = sampler.sample_dqm(dqm, self.time)

        return sampleset
    
    def solve_mock(
            self,
            params_inits: dict[str, Any] = None,
            hyper_optimizer: Optional[Optimizer] = None
    ) -> DiscreteQuadraticModel:
        """"
        mock solve method as a wrapper to Converter.to_cqm method
        created on 22.05 for learning purposes only 
        """
        return Converter.to_dqm_mock(self.problem)
