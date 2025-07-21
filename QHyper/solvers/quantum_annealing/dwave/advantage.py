import os
from typing import Any, Callable, Dict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from collections import defaultdict

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver, SolverResult, SamplesetData
from QHyper.converter import Converter
from QHyper.constraint import Polynomial

from dwave.system import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from dimod import BinaryQuadraticModel
from dwave.embedding.pegasus import find_clique_embedding
from minorminer import find_embedding
import warnings

from enum import Enum

import time
import dimod


DWAVE_API_TOKEN = os.environ.get("DWAVE_API_TOKEN")


class TimeUnits(str, Enum):
    S = "s"
    US = "us"


class Timing:
    FIND_CLIQUE_EMBEDDING = f"find_clique_embedding_time_{TimeUnits.S}"
    FIND_HEURISTIC_EMBEDDING = f"find_heuristic_embedding_time_{TimeUnits.S}"
    FIXED_EMBEDDING_COMPOSITE = f"fixed_embedding_composite_time_{TimeUnits.S}"
    SAMPLE_FUNCTION = f"sample_func_time_{TimeUnits.S}"


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

    def __init__(
        self,
        problem: Problem,
        penalty_weights: list[float] | None = None,
        num_reads: int = 1,
        chain_strength: float | None = None,
        use_clique_embedding: bool = False,
        token: str | None = None,
        elapse_times: bool = False,
        **config: Any,
    ) -> None:
        self.problem = problem
        self.penalty_weights = penalty_weights
        self.num_reads = num_reads
        self.chain_strength = chain_strength
        self.use_clique_embedding = use_clique_embedding
        self.sampler = DWaveSampler(token=token or DWAVE_API_TOKEN, **config)
        self.token = token
        self.elapse_times = elapse_times
        self.times: Dict = {}

        if use_clique_embedding:
            # args = self.weigths if self.weigths else []
            args = getattr(self, "weights", [])
            qubo = Converter.create_qubo(self.problem, args)
            qubo_terms, offset = convert_qubo_keys(qubo)
            bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

            self.embedding = execute_timed(
                lambda: find_clique_embedding(
                    bqm.to_networkx_graph(),
                    target_graph=self.sampler.to_networkx_graph(),
                ),
                self.elapse_times,
                self.times,
                Timing.FIND_CLIQUE_EMBEDDING,
            )

    def solve(
        self,
        penalty_weights: list[float] | None = None,
        return_metadata: bool = False,
    ) -> Any:
        if penalty_weights is None and self.penalty_weights is None:
            penalty_weights = [1.0] * (len(self.problem.constraints) + 1)
        penalty_weights = (
            self.penalty_weights if penalty_weights is None else penalty_weights
        )

        qubo = Converter.create_qubo(self.problem, penalty_weights)
        qubo_terms, offset = convert_qubo_keys(qubo)
        bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

        if not self.use_clique_embedding:
            self.embedding = execute_timed(
                lambda: find_embedding(
                    bqm.to_networkx_graph(),
                    self.sampler.to_networkx_graph(),
                ),
                self.elapse_times,
                self.times,
                Timing.FIND_HEURISTIC_EMBEDDING,
            )

        embedding_compose = execute_timed(
            lambda: FixedEmbeddingComposite(self.sampler, self.embedding),
            self.elapse_times,
            self.times,
            Timing.FIXED_EMBEDDING_COMPOSITE,
        )

        # Additional sampling info
        return_embedding = True

        # Resolving from the sampleset future-like object
        def _resolve_future_and_return(
            sampleset: dimod.sampleset.SampleSet,
        ) -> dimod.sampleset.SampleSet:
            sampleset.resolve()
            return sampleset

        sampleset: dimod.sampleset.SampleSet = execute_timed(
            lambda: _resolve_future_and_return(
                embedding_compose.sample(
                    bqm,
                    num_reads=self.num_reads,
                    chain_strength=self.chain_strength,
                    return_embedding=return_embedding,
                )
            ),
            self.elapse_times,
            self.times,
            Timing.SAMPLE_FUNCTION,
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

        if return_metadata and not sampleset.info["timing"]:
            warnings.warn(
                "No timing information available for the sampleset. ", UserWarning
            )

        if return_metadata:
            sampleset_info = SamplesetData(
                time_dict_to_ndarray(
                    add_time_units_to_dwave_timing_info(
                        sampleset.info["timing"], TimeUnits.US
                    )
                ),
                time_dict_to_ndarray(self.times),
            )
        else:
            sampleset_info = None

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
        return func()


def time_dict_to_ndarray(sampleset_info_times: dict[str, float]) -> np.ndarray:
    dtype = [(key, float) for key in sampleset_info_times]
    result = np.recarray((), dtype=dtype)
    for key, value in sampleset_info_times.items():
        setattr(result, key, value)

    return result


def add_time_units_to_dwave_timing_info(
    dwave_sampleset_info_timing: dict[str, float], time_unit: TimeUnits = TimeUnits.US
) -> dict[str, float]:
    """
    Add time units to the D-Wave timing info.

    Parameters:
    -----------
    dwave_sampleset_info_timing  : dict[str, float]
        DWave dictionary with timing information.
    time_unit : TimeUnits, optional
        The time unit to append to the keys (default by DWave docs is TimeUnits.US).

    Returns:
    --------
    np.ndarray
        A record array with the timing information and units.
    """
    dwave_keys_with_unit = [
        key + f"_{time_unit.value}" for key in dwave_sampleset_info_timing.keys()
    ]
    return dict(zip(dwave_keys_with_unit, dwave_sampleset_info_timing.values()))
