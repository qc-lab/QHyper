from typing import Any, Tuple, DefaultDict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult
from QHyper.converter import Converter
from QHyper.optimizers import (Optimizer, OptimizationResult)

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
from dimod.sampleset import SampleSet


@dataclass
class OptimizerFunction:
    embedding_compose: DWaveSampler
    problem: Problem

    def __call__(self, args: npt.NDArray) -> OptimizationResult:
        sampleset = run_advantage(self.problem, self.embedding_compose, args)
        result_energy = sampleset.first.energy

        return OptimizationResult(result_energy, args)

class Advantage(Solver):
    """
    Class for solving a problem using Advantage

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    region : str, default 'eu-central-1'
        The region in which D-Wave Advantage is available.
    """

    def __init__(self, problem: Problem, region: str = "eu-central-1",
                 optimizer: Optimizer | None = None) -> None:
        self.problem = problem
        self.region = region
        self.optimizer = optimizer

    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        sampler = DWaveSampler(solver='Advantage_system4.1')
        embedding_compose = EmbeddingComposite(sampler)

        opt_result = None
        if self.optimizer:
            args = (
                np.array(params_inits["weights"]).flatten() if params_inits["weights"]
                else None
            )

            opt_wrapper = OptimizerFunction(embedding_compose, self.problem)
            result = self.optimizer.minimize(opt_wrapper, args)
            opt_result = result.params

        qubo_arguments = opt_result if self.optimizer else params_inits.get("weights", [])
        sampleset = run_advantage(self.problem, embedding_compose, qubo_arguments)
        result = self.prepare_solver_result(sampleset.first.sample, qubo_arguments)

        return result

    def prepare_solver_result(self, result: defaultdict, arguments: npt.NDArray) -> SolverResult:
        sorted_keys = sorted(result.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
        values = ''.join(str(result[key]) for key in sorted_keys)
        probabilities = {values: 100.0}
        parameters = {values: arguments}

        return SolverResult(probabilities, parameters)

def run_advantage(problem: Problem, embedding_compose: DWaveSampler, args: npt.NDArray, ) -> SampleSet:
    qubo = Converter.create_qubo(problem, args)
    qubo_terms, offset = convert_qubo_keys(qubo.terms)
    bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)
    sampleset = embedding_compose.sample(bqm)

    return sampleset


def convert_qubo_keys(qubo: defaultdict) -> Tuple[DefaultDict[tuple, float], float]:
    new_qubo = defaultdict(float)
    offset = 0.0

    for k, v in qubo.items():
        if isinstance(k, tuple):
            if len(k) == 0:
                offset += v
                continue
            elif len(k) == 1:
                new_key = (k[0], k[0])
            else:
                new_key = k

            if new_key in new_qubo:
                new_qubo[new_key] += v
            else:
                new_qubo[new_key] = v
        else:
            raise ValueError("Keys in QUBO must be tuples")

    return (new_qubo, offset)
