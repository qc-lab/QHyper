from typing import Any
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult
from QHyper.converter import Converter
from QHyper.optimizers import (Optimizer, OptimizationResult)
from QHyper.constraint import Polynomial

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
from dimod.sampleset import SampleSet


@dataclass
class OptimizerFunction:
    embedding_compose: DWaveSampler
    problem: Problem

    def __call__(self, args: npt.NDArray) -> OptimizationResult:
        sampleset = self.run_advantage(args)
        result_energy = sampleset.first.energy

        return OptimizationResult(result_energy, args)

    def run_advantage(self, args: npt.NDArray) -> SampleSet:
        qubo = Converter.create_qubo(self.problem, args)
        qubo_terms, offset = convert_qubo_keys(qubo)
        bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)
        sampleset = self.embedding_compose.sample(bqm)

        return sampleset


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
        wrapper = OptimizerFunction(embedding_compose, self.problem)

        if self.optimizer:
            args = (
                np.array(params_inits["weights"]).flatten() if params_inits["weights"]
                else None
            )

            result = self.optimizer.minimize(wrapper, args)
            opt_result = result.params

        qubo_arguments = opt_result if self.optimizer else params_inits.get("weights", [])
        solutions = wrapper.run_advantage(qubo_arguments)

        result = np.recarray(
            (len(solutions),),
            dtype=([(v, int) for v in solutions.variables]
                   + [('probability', float)]
                   + [('energy', float)])
        )

        num_of_shots = solutions.record.num_occurrences.sum()
        for i, solution in enumerate(solutions.data()):
            for var in solutions.variables:
                result[var][i] = solution.sample[var]

            result['probability'][i] = (
                solution.num_occurrences / num_of_shots)
            result['energy'][i] = solution.energy

        return SolverResult(
            result,
            {},
            []
        )

    def prepare_solver_result(self, result: defaultdict, arguments: npt.NDArray) -> SolverResult:
        sorted_keys = sorted(result.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
        values = ''.join(str(result[key]) for key in sorted_keys)
        probabilities = {values: 100.0}
        parameters = {values: arguments}

        return SolverResult(probabilities, parameters)

def convert_qubo_keys(qubo: Polynomial) -> tuple[dict[tuple, float], float]:
    new_qubo = defaultdict(float)
    offset = 0.0

    qubo, offset = qubo.separate_const()
    for k, v in qubo.terms.items():
        if len(k) == 1:
            new_key = (k[0], k[0])
        elif len(k) > 2:
            raise ValueError("Only supports quadratic model")
        else:
            new_key = k

        new_qubo[new_key] += v

    return (new_qubo, offset)
