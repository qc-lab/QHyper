import numpy as np
import numpy.typing as npt
from typing import Any
from dataclasses import dataclass
from collections import defaultdict

from QHyper.devices.dwave import DWaveDevice
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult
from QHyper.converter import Converter
from QHyper.constraint import Polynomial

from dwave.system import EmbeddingComposite
from dwave.system.composites import FixedEmbeddingComposite
from dimod import BinaryQuadraticModel
from dwave.embedding.pegasus import find_clique_embedding


@dataclass
class Advantage(Solver):
    """
    Class for solving a problem using Advantage

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    device : DWaveDevice
        Configuration of the device the solver runs the problem on.
    penalty_weights : list[float] | None, default None
        Penalty weights used for converting Problem to QUBO. They connect
        cost function with constraints. If not specified, all penalty
        weights are set to 1.
    num_reads: int, default 1
        The number of times the solver is run.
    chain_strength: float or None, default None
        The coupling strength between qubits.
    use_clique_embedding: bool, default False
        Find clique for the embedding
    """

    problem: Problem
    device: DWaveDevice
    penalty_weights: list[float] | None = None
    num_reads: int = 1
    chain_strength: float | None = None
    use_clique_embedding: bool = False

    def __post_init__(self) -> None:
        self.sampler = self.device.make_sampler('advantage')

        if self.use_clique_embedding:
            penalty_weights = self.penalty_weights or []
            qubo = Converter.create_qubo(self.problem, penalty_weights)
            qubo_terms, offset = convert_qubo_keys(qubo)
            bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

            self.embedding = find_clique_embedding(
                bqm.to_networkx_graph(),
                target_graph=self.sampler.to_networkx_graph()
            )

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]
                    ) -> 'Advantage':
        config = dict(config)
        if 'device' in config:
            config['device'] = DWaveDevice.from_config(config['device'])
        return cls(problem, **config)

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
