import pennylane as qml
from pennylane import numpy as np

from typing import Any, Callable, cast

from dataclasses import dataclass, field

from QHyper.problems.base import Problem
from QHyper.optimizers import (
        OptimizationResult, Optimizer, Dummy, OptimizationParameter)

from QHyper.converter import Converter
from QHyper.polynomial import Polynomial
from QHyper.solvers.base import Solver, SolverResult

SIMULATOR_DEVICES = {
    'default.qubit', 'default.mixed', 'default.tensor',
    'lightning.qubit', 'lightning.gpu', 'lightning.kokkos', 'lightning.tensor',
    'qiskit.aer', 'qiskit.basicsim',
}
QISKIT_REMOTE = 'qiskit.remote'


@dataclass
class QAOA(Solver):
    """
    Clasic QAOA implementation.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
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
        Penalty weights used for converting Problem to QUBO. They connect cost function
        with constraints. If not specified, all penalty weights are set to 1.
    device : dict
        Device configuration (required). Accepted keys:

        - ``type`` -- ``'simulator'`` or ``'qpu'`` (required)
        - ``name`` -- PennyLane device name (required)

          - simulator: ``'default.qubit'``, ``'lightning.qubit'``,
            ``'qiskit.aer'``, ...
          - qpu: ``'qiskit.remote'`` (or another plugin)

        - ``backend`` -- hardware backend, required for ``'qiskit.remote'``
          (e.g. ``'melbourne'``), otherwise omit
    mixer : str
        Mixer name. Currently only 'pl_x_mixer' is supported.
    qubo_cache : dict[tuple[float, ...], qml.Hamiltonian]
        Cache for QUBO.
    dev : qml.devices.LegacyDevice
        PennyLane device instance.
    """

    problem: Problem
    layers: int
    gamma: OptimizationParameter
    beta: OptimizationParameter
    optimizer: Optimizer
    device: dict[str, Any]
    penalty_weights: list[float] | None = None
    backend: str = field(default="default.qubit", init=False)
    backend_name: str | None = field(default=None, init=False)
    mixer: str = "pl_x_mixer"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.devices.LegacyDevice | None = field(default=None, init=False)

    def __init__(
            self,
            problem: Problem,
            layers: int,
            gamma: OptimizationParameter,
            beta: OptimizationParameter,
            device: dict[str, Any],
            penalty_weights: list[float] | None = None,
            optimizer: Optimizer = Dummy(),
            mixer: str = "pl_x_mixer"
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.penalty_weights = penalty_weights
        self.layers = layers
        self.device = device
        self.backend, self.backend_name = self._parse_device(device)
        self.mixer = mixer
        self.qubo_cache = {}

    @staticmethod
    def _parse_device(device: dict[str, Any]) -> tuple[str, str | None]:
        if not isinstance(device, dict):
            raise ValueError("'device' must be a mapping")

        device_type = str(device.get('type', '')).lower()
        device_name = device.get('name')
        if not isinstance(device_name, str) or not device_name:
            raise ValueError("'device.name' must be a non-empty string")

        device_backend = device.get('backend')
        name_key = device_name.lower()

        if device_type == 'simulator':
            if name_key == QISKIT_REMOTE:
                raise ValueError(
                    "'qiskit.remote' is a remote QPU device, not a simulator. "
                    "Use 'qiskit.aer' for local Aer simulation, or device.type "
                    "'qpu' with a hardware backend"
                )
        elif device_type == 'qpu':
            if name_key == QISKIT_REMOTE and not device_backend:
                raise ValueError(
                    "'qiskit.remote' requires 'device.backend' "
                    "(e.g. a hardware backend like 'melbourne')"
                )
            if name_key in SIMULATOR_DEVICES:
                raise ValueError(
                    f"'{device_name}' is a simulator, use device.type 'simulator' "
                    "or 'qiskit.remote' with a backend for a real qpu"
                )
        else:
            raise ValueError(
                "'device.type' must be either 'simulator' or 'qpu'"
            )

        return device_name, (
            str(device_backend) if device_backend is not None else None)

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'QAOA':
        return cls(problem, **dict(config))

    def _get_num_of_wires(self) -> int:
        if self.dev is None:
            raise ValueError("Device not initialized")
        return len(self.dev.wires)

    def _make_device(self, wires: Any) -> qml.devices.LegacyDevice:
        kwargs = {'backend': self.backend_name} if self.backend_name else {}
        return qml.device(self.backend, wires=wires, **kwargs)

    def create_cost_operator(self, problem: Problem,
                             penalty_weights: list[float]
                             ) -> qml.Hamiltonian:
        if tuple(penalty_weights) not in self.qubo_cache:
            qubo = Converter.create_qubo(problem, penalty_weights)
            self.qubo_cache[tuple(
                penalty_weights)] = self._create_cost_operator(qubo)
        return self.qubo_cache[tuple(penalty_weights)]

    def _create_cost_operator(self, qubo: Polynomial) -> qml.Hamiltonian:
        result: qml.Hamiltonian | None = None
        const = 0

        for variables, coeff in qubo.terms.items():
            if not variables:
                const += coeff
                continue

            summand: qml.Hamiltonian | None = None
            for var in variables:
                if summand and str(var) in summand.wires:
                    continue
                encoded_var = cast(
                    qml.Hamiltonian,
                    0.5 * qml.Identity(str(var)) - 0.5 * qml.PauliZ(str(var))
                )

                summand = (summand @ encoded_var if summand
                           else coeff * encoded_var)
            result = result + summand if result else summand

        assert result is not None
        return (result + const * qml.Identity(result.wires[0])).simplify()

    def _hadamard_layer(self, cost_operator: qml.Hamiltonian) -> None:
        for i in cost_operator.wires:
            qml.Hadamard(str(i))

    def _create_mixing_hamiltonian(self, cost_operator: qml.Hamiltonian
                                   ) -> qml.Hamiltonian:
        return qml.qaoa.x_mixer([str(v) for v in cost_operator.wires])

    def _circuit(
        self,
        angles: list[float],
        cost_operator: qml.Hamiltonian,
    ) -> None:
        def qaoa_layer(gamma: list[float], beta: list[float]) -> None:
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(
                beta, self._create_mixing_hamiltonian(cost_operator))
        gamma, beta = angles[:len(angles) // 2], angles[len(angles) // 2:]
        self._hadamard_layer(cost_operator)
        qml.layer(qaoa_layer, self.layers, gamma, beta)

    def get_expval_circuit(
        self, penalty_weights: list[float]
    ) -> Callable[[list[float]], OptimizationResult]:
        cost_operator = self.create_cost_operator(self.problem, penalty_weights)
        self.dev = self._make_device(cost_operator.wires)

        @qml.qnode(self.dev)
        def expval_circuit(angles: list[float]) -> OptimizationResult:
            self._circuit(angles, cost_operator)
            return cast(float, qml.expval(cost_operator))

        def wrapper(angles: list[float]) -> OptimizationResult:
            return OptimizationResult(
                expval_circuit(angles), angles
            )

        return wrapper

    def get_probs_func(
        self, problem: Problem, penalty_weights: list[float]
    ) -> Callable[[list[float]], list[float]]:
        """Returns function that takes angles and returns probabilities

        Parameters
        ----------
        penalty_weights : list[float]
            Penalty weights for converting Problem to QUBO

        Returns
        -------
        Callable[[list[float]], float]
            Returns function that takes angles and returns probabilities
        """
        cost_operator = self.create_cost_operator(problem, penalty_weights)
        self.dev = self._make_device(cost_operator.wires)

        @qml.qnode(self.dev)
        def probability_circuit(angles: list[float]) -> list[float]:
            self._circuit(angles, cost_operator)
            return cast(
                list[float], qml.probs(wires=cost_operator.wires)
            )

        return probability_circuit

    def run_with_probs(
        self,
        problem: Problem,
        angles: list[float],
        penalty_weights: list[float],
    ) -> np.recarray:
        probs = self.get_probs_func(problem, penalty_weights)(angles)

        recarray = np.recarray((len(probs),),
                               dtype=[(wire, 'i4') for wire in
                                      self.dev.wires]+[('probability', 'f8')])
        for i, probability in enumerate(probs):
            solution = format(i, "b").zfill(self._get_num_of_wires())
            recarray[i] = *solution, probability
        return recarray

    def _run_optimizer(self, penalty_weights: list[float],
                       angles: OptimizationParameter) -> OptimizationResult:
        return self.optimizer.minimize(
            self.get_expval_circuit(penalty_weights), angles)

    def solve(self, penalty_weights: list[float] | None = None,
              gamma: list[float] | None = None,
              beta: list[float] | None = None) -> SolverResult:
        if penalty_weights is None and self.penalty_weights is None:
            penalty_weights = [1.] * (len(self.problem.constraints) + 1)
        penalty_weights = (self.penalty_weights if penalty_weights is None
                           else penalty_weights)

        assert penalty_weights is not None
        gamma_ = self.gamma if gamma is None else self.gamma.update(init=gamma)
        beta_ = self.beta if beta is None else self.beta.update(init=beta)

        angles = gamma_ + beta_
        opt_res = self._run_optimizer(penalty_weights, angles)

        gamma_res = opt_res.params[:len(opt_res.params) // 2]
        beta_res = opt_res.params[len(opt_res.params) // 2:]

        return SolverResult(
            self.run_with_probs(self.problem, opt_res.params, penalty_weights),
            {'gamma': gamma_res, 'beta': beta_res},
            opt_res.history,
        )
