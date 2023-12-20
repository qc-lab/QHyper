from typing import Any, Optional

from dimod import BinaryQuadraticModel

from QHyper.solvers.converter import Converter
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver

from dwave.system import DWaveSampler, EmbeddingComposite

class Advantage(Solver):
    """
    Class for solving a problem using
    Advantage 
    """

    def __init__(self, problem: Problem, time: float) -> None:
        self.problem: Problem = problem
        self.time: float = time

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        sampler = DWaveSampler(region="eu-central-1", solver='Advantage_system5.4')
        Q = self.problem.objective_function.dictionary
        sampleset = EmbeddingComposite(sampler).sample_qubo(Q)

        return sampleset

