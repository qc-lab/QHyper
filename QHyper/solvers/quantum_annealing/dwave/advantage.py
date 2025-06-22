import os
from typing import Any
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult
from QHyper.converter import Converter
from QHyper.constraint import Polynomial

from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system.composites import FixedEmbeddingComposite
from dimod import BinaryQuadraticModel
from dwave.embedding.pegasus import find_clique_embedding


DWAVE_API_TOKEN = os.environ.get('DWAVE_API_TOKEN')


@dataclass
class Advantage(Solver):
    """
    Class for solving a problem using Advantage

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    num_reads: int, default 1
        The number of times the solver is run.
    chain_strength: float or None, default None
        The coupling strength between qubits.
    hyper_optimizer: Optimizer or None, default None
        The optimizer for hyperparameters.
    params_inits: dict[str, Any], default {}
        The initial parameter settings.
    use_clique_embedding: bool, default False
        Find clique for the embedding
    **config: Any
        Config for the D-Wave solver. Documentation available at https://docs.dwavequantum.com 
    """

    problem: Problem
    penalty_weights: list[float] | None = None
    num_reads: int = 1
    chain_strength: float | None = None
    token: str | None = None

    def __init__(self,
                 problem: Problem,
                 penalty_weights: list[float] | None = None,
                 num_reads: int = 1,
                 chain_strength: float | None = None,
                 use_clique_embedding: bool = False,
                 token: str | None = None,
                 **config: Any) -> None:
        self.problem = problem
        self.penalty_weights = penalty_weights
        self.num_reads = num_reads
        self.chain_strength = chain_strength
        self.use_clique_embedding = use_clique_embedding
        self.sampler = DWaveSampler(
            token=token or DWAVE_API_TOKEN, **config)
        self.token = token

        if use_clique_embedding:
            args = self.weigths if self.weigths else []
            qubo = Converter.create_qubo(self.problem, args)
            qubo_terms, offset = convert_qubo_keys(qubo)
            bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

            self.embedding = find_clique_embedding(
                bqm.to_networkx_graph(),
                target_graph=self.sampler.to_networkx_graph()
            )

    def solve(self, penalty_weights: list[float] | None = None) -> Any:
        if penalty_weights is None and self.penalty_weights is None:
            penalty_weights = [1.] * (len(self.problem.constraints) + 1)
        penalty_weights = self.penalty_weights if penalty_weights is None else penalty_weights

        if not self.use_clique_embedding:
            embedding_compose = EmbeddingComposite(self.sampler)
        else:
            embedding_compose = FixedEmbeddingComposite(
                self.sampler, self.embedding)

        qubo = Converter.create_qubo(self.problem, penalty_weights)
        qubo_terms, offset = convert_qubo_keys(qubo)
        bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)
        sampleset = embedding_compose.sample(
            bqm, num_reads=self.num_reads, chain_strength=self.chain_strength
        )

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

        return SolverResult(result, {"penalty_weights": penalty_weights}, [])

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
