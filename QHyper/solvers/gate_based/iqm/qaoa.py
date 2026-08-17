from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from QHyper.problems.base import Problem
from QHyper.optimizers import (
    OptimizationResult,
    Optimizer,
    Dummy,
    OptimizationParameter,
    create_optimizer,
)
from QHyper.converter import Converter
from QHyper.polynomial import Polynomial
from QHyper.solvers.base import Solver, SolverResult
from QHyper.devices.iqm import IQMDevice


@dataclass
class QAOA(Solver):
    """
    QAOA implementation running on IQM quantum processors.

    Attributes
    ----------
    problem : Problem 
        The problem to be solved. 
    device : IQMDevice
        Configuration of the device the solver runs the problem on.
    layers : int
        Number of layers.
    gamma : OptimizationParameter
        Vector of gamma angles used in cost Hamiltonian. Size of the vector
        should be equal to the number of layers.
    beta : OptimizationParameter
        Vector of beta angles used in mixing Hamiltonian. Size of the vector
        should be equal to the number of layers.
    optimizer : Optimizer
        Optimizer used in the classical part of the algorithm.
    penalty_weights : list[float] | None
        Penalty weights used for converting Problem to QUBO. They connect cost
        function with constraints. If not specified, all penalty weights are
        set to 1.
    shots : int, default 1000
        Number of measurement shots per circuit execution.
    qubo_cache : dict[tuple[float, ...], tuple[SparsePauliOp, list[str]]]
        Cache mapping penalty weights to the cost operator and its variable
        names.
    """

    problem: Problem
    device: IQMDevice
    layers: int
    gamma: OptimizationParameter
    beta: OptimizationParameter
    optimizer: Optimizer = Dummy()
    penalty_weights: list[float] | None = None
    shots: int = 1000
    qubo_cache: dict[tuple[float, ...], tuple[SparsePauliOp, list[str]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if self.device is None:
            raise ValueError("'solver.device' is required for IQM solver")
        self._backend = self.device.make_backend()

        if self.layers <= 0:
            raise ValueError("layers must be >= 1")
        if len(self.gamma) != self.layers or len(self.beta) != self.layers:
            warnings.warn(
                f"Length of gamma ({len(self.gamma)}) or beta "
                f"({len(self.beta)}) does not match the number of "
                f"layers ({self.layers}).",
                UserWarning,
            )

    def create_cost_operator(
        self, problem: Problem, penalty_weights: list[float]
    ) -> tuple[SparsePauliOp, list[str]]:
        key = tuple(penalty_weights)
        if key not in self.qubo_cache:
            qubo = Converter.create_qubo(problem, penalty_weights)
            self.qubo_cache[key] = self._create_cost_operator(qubo)
        return self.qubo_cache[key]

    def _create_cost_operator(
        self, qubo: Polynomial
    ) -> tuple[SparsePauliOp, list[str]]:
        var_names = sorted(
            {str(v) for term in qubo.terms for v in term if v is not None}
        )
        n = len(var_names)
        name_to_idx = {name: i for i, name in enumerate(var_names)}

        coeffs: dict[str, float] = {}
        const = 0.0

        for variables, coeff in qubo.terms.items():
            if not variables:
                const += coeff
                continue

            idxs = list(dict.fromkeys(name_to_idx[str(v)] for v in variables))

            m = len(idxs)
            for mask in range(1 << m):
                z_positions = {idxs[bit] for bit in range(m) if mask & (1 << bit)}
                sign = (-0.5) ** len(z_positions) * 0.5 ** (m - len(z_positions))
                value = coeff * sign

                if not z_positions:
                    const += value
                else:
                    label = "".join("Z" if j in z_positions else "I" for j in range(n))
                    coeffs[label] = coeffs.get(label, 0.0) + value

        if abs(const) > 0.0:
            label_I = "I" * max(1, n)
            coeffs[label_I] = coeffs.get(label_I, 0.0) + const

        if not coeffs:
            op = SparsePauliOp.from_list([("I" * max(1, n), 0.0)])
        else:
            op = SparsePauliOp.from_list(list(coeffs.items()))

        return op.simplify(), var_names

    def _hadamard_layer(self, qc: QuantumCircuit) -> None:
        for q in range(qc.num_qubits):
            qc.h(q)

    def _cost_layer(
        self, qc: QuantumCircuit, cost_op: SparsePauliOp, gamma: float
    ) -> None:
        qc.append(PauliEvolutionGate(cost_op, time=float(gamma)), qc.qubits)

    def _mixer_layer(self, qc: QuantumCircuit, beta: float) -> None:
        for q in range(qc.num_qubits):
            qc.rx(2.0 * float(beta), q)

    def _circuit(self, cost_op: SparsePauliOp, angles: list[float]) -> QuantumCircuit:
        gamma, beta = angles[: len(angles) // 2], angles[len(angles) // 2 :]
        n = cost_op.num_qubits
        qc = QuantumCircuit(n, name=f"QAOA_p{self.layers}")

        self._hadamard_layer(qc)
        for layer in range(self.layers):
            self._cost_layer(qc, cost_op, gamma[layer])
            self._mixer_layer(qc, beta[layer])

        return qc

    def _clean_bitstring(self, key: str, n: int) -> str:
        s = "".join(ch for ch in str(key) if ch in "01")
        return s.zfill(n)[-n:]

    def _run_circuit(self, qc: QuantumCircuit) -> dict[str, int]:
        n = qc.num_qubits
        qc_meas = qc.copy()
        qc_meas.add_register(ClassicalRegister(n, "c"))
        qc_meas.measure(range(n), range(n))
        tqc = self.device.transpile(qc_meas, self._backend)
        return self._backend.run(tqc, shots=self.shots).result().get_counts()

    def _expval_from_counts(
        self, counts: dict[str, int], cost_op: SparsePauliOp
    ) -> float:
        total = max(1, sum(counts.values()))
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
                acc += parity * (freq / total)
            expval += c * acc

        return float(expval)

    def get_expval_circuit(
        self, penalty_weights: list[float]
    ) -> Callable[[list[float]], OptimizationResult]:
        cost_op, _ = self.create_cost_operator(self.problem, penalty_weights)

        def wrapper(angles: list[float]) -> OptimizationResult:
            qc = self._circuit(cost_op, angles)
            counts = self._run_circuit(qc)
            val = self._expval_from_counts(counts, cost_op)
            return OptimizationResult(val, list(angles))

        return wrapper

    def get_probs_func(
        self, problem: Problem, penalty_weights: list[float]
    ) -> Callable[[list[float]], list[float]]:
        cost_op, _ = self.create_cost_operator(problem, penalty_weights)

        def probability_fn(angles: list[float]) -> list[float]:
            qc = self._circuit(cost_op, angles)
            counts = self._run_circuit(qc)
            n = cost_op.num_qubits
            total = max(1, sum(counts.values()))

            probs = [0.0] * (1 << n)
            for key, freq in counts.items():
                s = self._clean_bitstring(key, n)
                probs[int(s, 2)] += freq / total
            return probs

        return probability_fn

    def run_with_probs(
        self,
        problem: Problem,
        angles: list[float],
        penalty_weights: list[float],
    ) -> np.recarray:
        probs = self.get_probs_func(problem, penalty_weights)(angles)
        _, var_names = self.create_cost_operator(problem, penalty_weights)
        n = len(var_names)

        recarray = np.recarray(
            (len(probs),),
            dtype=[(name, "i4") for name in var_names] + [("probability", "f8")],
        )
        for i, probability in enumerate(probs):
            solution = format(i, "b").zfill(n)
            recarray[i] = *solution, probability
        return recarray

    def _run_optimizer(
        self, penalty_weights: list[float], angles: OptimizationParameter
    ) -> OptimizationResult:
        return self.optimizer.minimize(self.get_expval_circuit(penalty_weights), angles)

    def solve(
        self,
        penalty_weights: list[float] | None = None,
        gamma: list[float] | None = None,
        beta: list[float] | None = None,
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

        gamma_res = opt_res.params[: len(opt_res.params) // 2]
        beta_res = opt_res.params[len(opt_res.params) // 2 :]

        return SolverResult(
            self.run_with_probs(self.problem, opt_res.params, penalty_weights),
            {"gamma": gamma_res, "beta": beta_res},
            opt_res.history,
        )

    @classmethod
    def from_config(cls, problem: Problem,
                    config: dict[str, Any]) -> 'QAOA':
        config = dict(config)

        for param_name in ('gamma', 'beta'):
            if param_name in config and isinstance(config[param_name], dict):
                config[param_name] = OptimizationParameter(
                    **config[param_name])

        if 'optimizer' in config and isinstance(config['optimizer'], dict):
            config['optimizer'] = create_optimizer(config['optimizer'])

        if 'device' not in config:
            raise ValueError("'solver.device' is required for IQM solver")
        config['device'] = IQMDevice.from_config(config['device'])

        return cls(problem, **config)
