from typing import Any
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult, SolverConfigException
from QHyper.converter import Converter
from QHyper.constraint import Polynomial
from QHyper.util import weighted_avg_evaluation
from QHyper.optimizers import (
    OPTIMIZERS, Optimizer, OptimizationResult)

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
from dimod.sampleset import SampleSet


@dataclass
class OptimizerFunction:
    embedding_compose: DWaveSampler
    problem: Problem
    num_reads: int
    chain_strength: float | None

    def __call__(self, args: npt.NDArray) -> OptimizationResult:
        sampleset = self.run_advantage(args)

        result = np.recarray(
            (len(sampleset),),
            dtype=([(v, int) for v in sampleset.variables]
                   + [('probability', float)]
                   + [('energy', float)])
        )

        num_of_shots = sampleset.record.num_occurrences.sum()
        for i, solution in enumerate(sampleset.data()):
            for var in sampleset.variables:
                result[var][i] = solution.sample[var]

            result['probability'][i] = (
                solution.num_occurrences / num_of_shots)
            result['energy'][i] = solution.energy

        result_energy = weighted_avg_evaluation(
            result, self.problem.get_score, penalty=0
        )

        return OptimizationResult(result_energy, args)

    def run_advantage(self, args: npt.NDArray) -> SampleSet:
        qubo = Converter.create_qubo(self.problem, args)
        qubo_terms, offset = convert_qubo_keys(qubo)
        bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)
        sampleset = self.embedding_compose.sample(
            bqm, num_reads=self.num_reads, chain_strength=self.chain_strength
        )

        return sampleset


class Advantage(Solver):
    """
    Class for solving a problem using Advantage

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    version: str, default 'Advantage_system5.4'
        The version of the D-Wave Advantage system.
    region : str, default 'eu-central-1'
        The region in which D-Wave Advantage is available.
    num_reads: int, default 1
        The number of times the solver is run.
    chain_strength: float or None, default None
        The coupling strength between qubits.
    hyper_optimizer: Optimizer or None, default None
        The optimizer for hyperparameters.
    params_inits: dict[str, Any] | None, default None
        The initial parameter settings.
    """

    def __init__(self, problem: Problem,
                 version: str = "Advantage_system5.4",
                 region: str = "eu-central-1",
                 num_reads: int = 1,
                 chain_strength: float | None = None,
                 hyper_optimizer: Optimizer | None = None,
                 params_inits: dict[str, Any] | None = None) -> None:
        self.problem = problem
        self.version = version
        self.region = region
        self.num_reads = num_reads
        self.chain_strength = chain_strength
        self.hyper_optimizer = hyper_optimizer
        self.params_inits = params_inits

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'Advantage':
        if not (hyper_optimizer_config := config.pop('hyper_optimizer', None)):
            hyper_optimizer = None
        elif not (hyper_optimizer_type := hyper_optimizer_config.pop('type', None)):
            raise SolverConfigException(
                "Optimizer type was not provided")
        elif not (hyper_optimizer_class := OPTIMIZERS.get(
                hyper_optimizer_type, None)):
            raise SolverConfigException(
                f"There is no {hyper_optimizer_type} optimizer type")
        else:
            hyper_optimizer = hyper_optimizer_class(**hyper_optimizer_config)

        version = config.pop('version', "Advantage_system5.4")
        region = config.pop('region', "eu-central-1")
        num_reads = config.pop('num_reads', 1)
        chain_strength = config.pop('chain_strength', None)

        params_inits = config.pop('params_inits', None)

        return cls(problem, version, region, num_reads, chain_strength, hyper_optimizer, params_inits)


    def solve(self, params_inits: dict[str, Any] = {}) -> Any:
        if not params_inits:
            params_inits = self.params_inits
        sampler = DWaveSampler(solver=self.version, region=self.region)
        embedding_compose = EmbeddingComposite(sampler)

        opt_result = None
        wrapper = OptimizerFunction(
            embedding_compose, self.problem, self.num_reads, self.chain_strength
        )

        if self.hyper_optimizer:
            args = (
                np.array(params_inits["weights"]).flatten() if params_inits["weights"]
                else None
            )

            result = self.hyper_optimizer.minimize(wrapper, args)
            opt_result = result.params

        qubo_arguments = opt_result if self.hyper_optimizer else params_inits.get("weights", [])
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
            {"weights": qubo_arguments},
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
