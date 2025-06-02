import os
from typing import Any, Callable, Dict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult, SamplesetInfo
from QHyper.converter import Converter
from QHyper.constraint import Polynomial

from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system.composites import FixedEmbeddingComposite
from dimod import BinaryQuadraticModel
from dwave.embedding.pegasus import find_clique_embedding

import time


DWAVE_API_TOKEN = os.environ.get("DWAVE_API_TOKEN")


@dataclass
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
    params_inits: dict[str, Any], default {}
        The initial parameter settings.
    use_clique_embedding: bool, default False
        Find clique for the embedding
    """

    problem: Problem
    penalty_weights: list[float] | None = None
    version: str = "Advantage_system5.4"
    region: str = "eu-central-1"
    num_reads: int = 1
    chain_strength: float | None = None
    token: str | None = None

    def __init__(
        self,
        problem: Problem,
        penalty_weights: list[float] | None = None,
        version: str = "Advantage_system5.4",
        region: str = "eu-central-1",
        num_reads: int = 1,
        chain_strength: float | None = None,
        use_clique_embedding: bool = False,
        token: str | None = None,
        measure_times: bool = False,
    ) -> None:
        self.problem = problem
        self.penalty_weights = penalty_weights
        self.version = version
        self.region = region
        self.num_reads = num_reads
        self.chain_strength = chain_strength
        self.use_clique_embedding = use_clique_embedding
        self.sampler = DWaveSampler(
            solver=self.version,
            region=self.region,
            token=token or DWAVE_API_TOKEN,
        )
        self.token = token
        self.measure_times = measure_times
        self.times: Dict = {}

        if use_clique_embedding:
            # args = self.weigths if aself.weigths else []
            args = getattr(self, "weights", [])
            qubo = Converter.create_qubo(self.problem, args)
            qubo_terms, offset = convert_qubo_keys(qubo)
            bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

            find_clique_emb_handler = find_clique_embedding(
                bqm.to_networkx_graph(),
                target_graph=self.sampler.to_networkx_graph(),
            )
            self.embedding = execute_timed(
                find_clique_emb_handler,
                measure_times,
                self.times,
                "find_clique_embedding_time",
            )

    def solve(
        self,
        penalty_weights: list[float] | None = None,
        return_sampleset_info: bool = False,
    ) -> Any:
        if penalty_weights is None and self.penalty_weights is None:
            penalty_weights = [1.0] * (len(self.problem.constraints) + 1)
        penalty_weights = (
            self.penalty_weights if penalty_weights is None else penalty_weights
        )

        if not self.use_clique_embedding:
            embedding_compose = execute_timed(
                EmbeddingComposite(self.sampler),
                self.measure_times,
                self.times,
                "embedding_composite_time",
            )
        else:
            embedding_compose = execute_timed(
                FixedEmbeddingComposite(self.sampler, self.embedding),
                self.measure_times,
                self.times,
                "fixed_embedding_composite_time",
            )

        qubo = Converter.create_qubo(self.problem, penalty_weights)
        qubo_terms, offset = convert_qubo_keys(qubo)
        bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

        # Additional sampling info
        return_embedding = True
        sample_func_handler = embedding_compose.sample(
            bqm,
            num_reads=self.num_reads,
            chain_strength=self.chain_strength,
            return_embedding=return_embedding,
        )
        sampleset = execute_timed(
            sample_func_handler, self.measure_times, self.times, "sample_time"
        )

        result = np.recarray(
            (len(sampleset),),
            dtype=(
                [(v, int) for v in sampleset.variables]
                + [("probability", float)]
                + [("energy", float)]
            ),
        )

        num_of_shots = sampleset.record.num_occurrences.sum()
        for i, solution in enumerate(sampleset.data()):
            for var in sampleset.variables:
                result[var][i] = solution.sample[var]

            result["probability"][i] = solution.num_occurrences / num_of_shots
            result["energy"][i] = solution.energy

        sampleset_info = None
        if return_sampleset_info:
            sampleset_info = SamplesetInfo(
                time_dict_to_ndarray(sampleset.info["timing"]),
                time_dict_to_ndarray(self.times),
            )

        return SolverResult(
            result, {"penalty_weights": penalty_weights}, [], sampleset_info
        )

    def prepare_solver_result(
        self, result: defaultdict, arguments: npt.NDArray
    ) -> SolverResult:
        sorted_keys = sorted(
            result.keys(), key=lambda x: int("".join(filter(str.isdigit, x)))
        )
        values = "".join(str(result[key]) for key in sorted_keys)
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


def execute_timed(
    func: Callable, measure_time: bool, times_dict: Dict, key: str
) -> Any:
    """
    Execute a function with optional timing measurement.

    Parameters:
    -----------
    func : callable
        The function to execute and potentially time.
    measure_time : bool
        Whether to measure execution time.
    times_dict : dict
        Dictionary where timing results will be stored.
    key : str
        Key to use for storing the timing result.
    Returns:
    --------
    The result of func executed.
    """
    if measure_time:
        start_time = time.perf_counter()
        result = func()
        times_dict[key] = time.perf_counter() - start_time
        return result
    else:
        return func()  # Execute without timing overhead


def time_dict_to_ndarray(sampleset_info_times: dict[str, float]) -> np.ndarray:
    dtype = [(key, float) for key in sampleset_info_times]
    result = np.recarray((), dtype=dtype)
    for key, value in sampleset_info_times.items():
        setattr(result, key, value)

    return result
