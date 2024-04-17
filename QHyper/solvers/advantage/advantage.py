from typing import Any

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.converter import Converter

from dwave.system import DWaveSampler, EmbeddingComposite

class Advantage(Solver):
    """
    Class for solving a problem using
    Advantage 
    """

    def __init__(self, problem: Problem) -> None:
        self.problem: Problem = problem

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        sampler = DWaveSampler(region="eu-central-1", solver='Advantage_system5.4')
        Q = Converter.create_weight_free_qubo(self.problem)
        sampleset = EmbeddingComposite(sampler).sample_qubo(Q)

        return sampleset

