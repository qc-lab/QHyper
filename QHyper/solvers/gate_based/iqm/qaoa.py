from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple, Optional
import os
import time
import numpy as np
import warnings

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from iqm.qiskit_iqm import IQMProvider

from QHyper.problems.base import Problem
from QHyper.optimizers import (
    OptimizationResult,
    Optimizer,
    Dummy,
    OptimizationParameter,
)
from QHyper.converter import Converter
from QHyper.polynomial import Polynomial
from QHyper.solvers.base import Solver, SolverResult


@dataclass
class QAOA(Solver):
    problem: Problem
    layers: int
    gamma: OptimizationParameter
    beta: OptimizationParameter
    optimizer: Optimizer = Dummy()
    penalty_weights: Optional[List[float]] = None

    backend_url: str = "https://cocos.resonance.meetiqm.com/garnet"
    backend_token: Optional[str] = None
    shots: int = 1000

    qubo_cache: Dict[Tuple[float, ...], Tuple[SparsePauliOp, List[str]]] = field(
        default_factory=dict, init=False
    )
    _backend = None

    def __post_init__(self):
        if self.backend_token is not None:
            os.environ["IQM_TOKEN"] = self.backend_token
        provider = IQMProvider(url=self.backend_url)
        self._backend = provider.get_backend()

        if self.layers <= 0:
            raise ValueError("layers must be >= 1")
        if len(self.gamma) != self.layers or len(self.beta) != self.layers:
            warnings.warn(
                f"Length of gamma ({len(self.gamma)}) or beta ({len(self.beta)}) "
                f"does not match the number of layers ({self.layers}).",
                UserWarning,
            )

    def _clean_bitstring(self, key, n: int) -> str:
        s = "".join(ch for ch in str(key) if ch in "01")
        if len(s) < n:
            return s.zfill(n)
        if len(s) > n:
            return s[-n:]
        return s

    def _get_cost_operator(
        self, penalty_weights: List[float]
    ) -> Tuple[SparsePauliOp, List[str]]:
        key = tuple(float(x) for x in penalty_weights)
        if key not in self.qubo_cache:
            qubo: Polynomial = Converter.create_qubo(self.problem, penalty_weights)

            var_names = sorted(
                {str(v) for term in qubo.terms.keys() for v in term if v is not None}
            )
            n = len(var_names)
            name_to_idx = {name: i for i, name in enumerate(var_names)}

            coeffs: Dict[str, float] = {}
            const = 0.0

            def _all_subsets(items: Sequence[int]):
                yield ()
                m = len(items)
                for r in range(1, m + 1):

                    def rec(start: int, left: int, acc: List[int]):
                        if left == 0:
                            yield tuple(acc)
                            return
                        for j in range(start, m - left + 1):
                            acc.append(items[j])
                            yield from rec(j + 1, left - 1, acc)
                            acc.pop()

                    yield from rec(0, r, [])

            for variables, coeff in qubo.terms.items():
                if not variables:
                    const += coeff
                    continue
                idxs = [name_to_idx[str(v)] for v in variables]
                m = len(idxs)
                norm = (0.5) ** m
                for subset in _all_subsets(idxs):
                    sign = (-1.0) ** (len(subset))
                    if len(subset) == 0:
                        const += coeff * norm
                    else:
                        zset = set(subset)
                        label = "".join("Z" if j in zset else "I" for j in range(n))
                        coeffs[label] = coeffs.get(label, 0.0) + coeff * sign * norm

            if abs(const) > 0.0:
                label_I = "I" * max(1, n)
                coeffs[label_I] = coeffs.get(label_I, 0.0) + const

            if not coeffs:
                op = SparsePauliOp.from_list([("I" * max(1, n), 0.0)])
            else:
                labels, values = zip(*coeffs.items())
                op = SparsePauliOp.from_list(list(zip(labels, values)))

            op = op.simplify()
            self.qubo_cache[key] = (op, var_names)
        return self.qubo_cache[key]

    def _hadamards(self, qc: QuantumCircuit):
        for q in range(qc.num_qubits):
            qc.h(q)

    def _cost_layer(self, qc: QuantumCircuit, cost_op: SparsePauliOp, gamma: float):
        qc.append(PauliEvolutionGate(cost_op, time=float(gamma)), qc.qubits)

    def _mixer_layer(self, qc: QuantumCircuit, beta: float):
        for q in range(qc.num_qubits):
            qc.rx(2.0 * float(beta), q)

    def _build_qaoa_circuit(
        self, cost_op: SparsePauliOp, angles: Sequence[float]
    ) -> QuantumCircuit:
        p = int(self.layers)
        assert len(angles) == 2 * p, f"angles length {len(angles)} != 2*layers ({2*p})"
        gamma = angles[:p]
        beta = angles[p:]

        n = cost_op.num_qubits
        qc = QuantumCircuit(n, name=f"QAOA_p{p}")
        self._hadamards(qc)
        for l in range(p):
            self._cost_layer(qc, cost_op, gamma[l])
            self._mixer_layer(qc, beta[l])
        return qc

    def _counts(self, qc: QuantumCircuit) -> Dict[str, int]:
        n = qc.num_qubits
        qc_meas = qc.copy()
        creg = ClassicalRegister(n, "c")
        qc_meas.add_register(creg)
        qc_meas.measure(range(n), range(n))

        tqc = transpile(qc_meas, backend=self._backend, optimization_level=3)
        job = self._backend.run(tqc, shots=self.shots)
        res = job.result()
        counts = res.get_counts()
        return counts

    def _exp_from_counts(self, counts: Dict[str, int], cost_op: SparsePauliOp) -> float:
        shots = max(1, sum(counts.values()))
        n = cost_op.num_qubits
        expval = 0.0
        for label, coeff in cost_op.to_list():
            c = float(np.real(coeff))
            if label == "I" * n:
                expval += c
                continue

            acc = 0.0
            for bitstr, freq in counts.items():
                s = self._clean_bitstring(bitstr, n)
                parity = 1
                for qi in range(n):
                    if label[qi] == "Z" and s[-1 - qi] == "1":
                        parity *= -1
                acc += parity * (freq / shots)
            expval += c * acc
        return float(expval)

    def _expectation(self, qc: QuantumCircuit, cost_op: SparsePauliOp) -> float:
        counts = self._counts(qc)
        return self._exp_from_counts(counts, cost_op)

    def get_expval_circuit(
        self, penalty_weights: List[float]
    ) -> Callable[[List[float]], OptimizationResult]:
        cost_op, _ = self._get_cost_operator(penalty_weights)

        def f(angles: List[float]) -> OptimizationResult:
            qc = self._build_qaoa_circuit(cost_op, angles)
            val = self._expectation(qc, cost_op)
            params = [float(np.asarray(v)) for v in angles]
            return OptimizationResult(val, params)

        return f

    def get_probs_func(
        self, problem: Problem, penalty_weights: List[float]
    ) -> Callable[[List[float]], List[float]]:
        cost_op, _ = self._get_cost_operator(penalty_weights)

        def probs_fn(angles: List[float]) -> List[float]:
            qc = self._build_qaoa_circuit(cost_op, angles)
            counts = self._counts(qc)
            n = cost_op.num_qubits
            size = 1 << n
            total = max(1, sum(counts.values()))

            probs = [0.0] * size
            for key, c in counts.items():
                s = self._clean_bitstring(key, n)
                idx = int(s, 2)
                probs[idx] += c / total
            return probs

        return probs_fn

    def run_with_probs(
        self, problem: Problem, angles: List[float], penalty_weights: List[float]
    ) -> np.recarray:
        probs = self.get_probs_func(problem, penalty_weights)(angles)
        cost_op, var_names = self._get_cost_operator(penalty_weights)
        n = cost_op.num_qubits

        rec = np.recarray(
            (len(probs),),
            dtype=[(name, "i4") for name in var_names] + [("probability", "f8")],
        )
        for i, p in enumerate(probs):
            bits = format(i, f"0{n}b")
            rec[i] = (*[int(b) for b in bits], float(p))
        return rec

    def _run_optimizer(
        self, penalty_weights: List[float], angles: OptimizationParameter
    ) -> OptimizationResult:
        return self.optimizer.minimize(self.get_expval_circuit(penalty_weights), angles)

    def solve(
        self,
        penalty_weights: Optional[List[float]] = None,
        gamma: Optional[List[float]] = None,
        beta: Optional[List[float]] = None,
    ) -> SolverResult:

        if penalty_weights is None and self.penalty_weights is None:
            penalty_weights = [1.0] * (len(self.problem.constraints) + 1)
        penalty_weights = (
            self.penalty_weights if penalty_weights is None else penalty_weights
        )

        gamma_ = self.gamma if gamma is None else self.gamma.update(init=gamma)
        beta_ = self.beta if beta is None else self.beta.update(init=beta)
        angles = gamma_ + beta_

        opt_res = self._run_optimizer(penalty_weights, angles)
        p = int(self.layers)
        gamma_res = opt_res.params[:p]
        beta_res = opt_res.params[p:]

        return SolverResult(
            self.run_with_probs(self.problem, opt_res.params, penalty_weights),
            {"gamma": gamma_res, "beta": beta_res},
            opt_res.history,
        )
