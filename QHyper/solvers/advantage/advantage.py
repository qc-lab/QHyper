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

    def __init__(self, problem: Problem, region: str = "eu-central-1") -> None:
        self.problem = problem
        self.region = region

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        sampler = DWaveSampler(region=self.region, solver='Advantage_system5.4')
        qubo = Converter.create_qubo(self.problem, params_inits.get("weights", []))
        sampleset = EmbeddingComposite(sampler).sample_qubo(qubo.terms)

        return sampleset

