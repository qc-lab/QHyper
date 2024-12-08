import pennylane as qml
from pennylane import numpy as np

from numpy.typing import NDArray
from typing import Any, Callable, cast

from dataclasses import dataclass, field

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult, Optimizer, Dummy, OptimizationParameter

from QHyper.converter import Converter
from QHyper.polynomial import Polynomial
from QHyper.solvers.base import Solver, SolverResult, SolverException


@dataclass
class QAOA(Solver):
    """
    Clasic QAOA implementation.

    Attributes
    ----------
    layers : int
        Number of layers.
    backend : str
        Backend for PennyLane.
    mixer : str
        Mixer name.
    qubo_cache : dict[tuple[float, ...], qml.Hamiltonian]
        Cache for QUBO.
    dev : qml.Device
        PennyLane device instance.
    """

    problem: Problem
    optimizer: Optimizer
    gamma: OptimizationParameter
    beta: OptimizationParameter
    weights: list[float] | None
    layers: int = 3
    backend: str = "default.qubit"
    mixer: str = "pl_x_mixer"
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = field(
        default_factory=dict, init=False)
    dev: qml.Device | None = field(default=None, init=False)

    def __init__(
            self, problem: Problem,
            gamma: OptimizationParameter,
            beta: OptimizationParameter,
            weights: list[float] | None = None,
            optimizer: Optimizer = Dummy(),
            layers: int = 3, backend: str = "default.qubit",
            mixer: str = "pl_x_mixer"
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.weights = weights
        self.layers = layers
        self.backend = backend
        self.mixer = mixer
        self.qubo_cache = {}

    def _get_num_of_wires(self) -> int:
        if self.dev is None:
            raise ValueError("Device not initialized")
        return len(self.dev.wires)

    def create_cost_operator(self, problem: Problem, weights: list[float]
                             ) -> qml.Hamiltonian:
        if tuple(weights) not in self.qubo_cache:
            qubo = Converter.create_qubo(problem, weights)
            self.qubo_cache[tuple(weights)] = self._create_cost_operator(qubo)
        return self.qubo_cache[tuple(weights)]

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
        self, weights: list[float]
    ) -> Callable[[list[float]], OptimizationResult]:
        cost_operator = self.create_cost_operator(self.problem, weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

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
        self, problem: Problem, weights: list[float]
    ) -> Callable[[list[float]], list[float]]:
        """Returns function that takes angles and returns probabilities

        Parameters
        ----------
        weights : list[float]
            weights for converting Problem to QUBO

        Returns
        -------
        Callable[[list[float]], float]
            Returns function that takes angles and returns probabilities
        """
        cost_operator = self.create_cost_operator(problem, weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

        @qml.qnode(self.dev)
        def probability_circuit(angles: list[float]) -> list[float]:
            self._circuit(angles, cost_operator)
            return cast(
                list[float], qml.probs(wires=cost_operator.wires)
            )

        return probability_circuit

    # def run_opt(
    #     self,
    #     problem: Problem,
    #     opt_args: NDArray,
    #     hyper_args: NDArray,
    # ) -> OptimizationResult:
    #     results = self.get_expval_circuit(problem, hyper_args)(opt_args)
    #     return OptimizationResult(results, opt_args)
    #
    def run_with_probs(
        self,
        problem: Problem,
        angles: list[float],
        weights: list[float],
    ) -> np.recarray:
        probs = self.get_probs_func(problem, weights)(angles)

        recarray = np.recarray((len(probs),),
                               dtype=[(wire, 'i4') for wire in
                                      self.dev.wires]+[('probability', 'f8')])
        for i, probability in enumerate(probs):
            solution = format(i, "b").zfill(self._get_num_of_wires())
            recarray[i] = *solution, probability
        return recarray

    # def get_opt_args(
    #     self,
    #     params_init: dict[str, Any],
    #     args: Optional[NDArray] = None,
    #     hyper_args: Optional[NDArray] = None,
    # ) -> NDArray:
    #     return np.array(
    #         args if args is not None
    #         else np.array(params_init["angles"])
    #     ).flatten()
    #
    # def get_hopt_args(
    #     self,
    #     params_init: dict[str, Any],
    #     args: Optional[NDArray] = None,
    #     hyper_args: Optional[NDArray] = None,
    # ) -> NDArray:
    #     return (
    #         hyper_args
    #         if hyper_args is not None
    #         else np.array(params_init["hyper_args"])
    #     )

    # def get_params_init_format(
    #     self, opt_args: NDArray, hyper_args: NDArray
    # ) -> dict[str, Any]:
    #     return {
    #         "angles": opt_args,
    #         "hyper_args": hyper_args,
    #     }

    def _run_optimizer(self, weights: list[float],
                       angles: OptimizationParameter) -> OptimizationResult:
        return self.optimizer.minimize(
            self.get_expval_circuit(weights), angles)

    def solve(self, weights: list[float] | None = None,
              gamma: list[float] | None = None,
              beta: list[float] | None = None) -> SolverResult:
        # if gamma is None and self.gamma is None:
        #     raise SolverException("Parameter 'gamma' was not provided")
        # if beta is None and self.beta is None:
        #     raise SolverException("Parameter 'beta' was not provided")

        # gamma = self.gamma if gamma is None else gamma
        # beta = self.beta if beta is None else beta
        if weights is None and self.weights is None:
            weights = [1.] * (len(self.problem.constraints) + 1)
        weights = self.weights if weights is None else weights

        gamma_ = self.gamma if gamma is None else self.gamma.update(init=gamma)
        beta_ = self.beta if beta is None else self.beta.update(init=beta)

        # opt_wrapper = LocalOptimizerFunction(
        #         self.pqc, self.problem, best_hargs)
        # opt_res = self.optimizer.minimize(opt_wrapper, opt_args)
        # func = self.get_expval_circuit(weights)

        # opt_res = self.optimizer.minimize(func, angles)
        # assert gamma
        # assert beta
        angles = gamma_ + beta_
        opt_res = self._run_optimizer(weights, angles)

        gamma_res = opt_res.params[:len(opt_res.params) // 2]
        beta_res = opt_res.params[len(opt_res.params) // 2:]

        return SolverResult(
            self.run_with_probs(self.problem, opt_res.params, weights),
            {'gamma': gamma_res, 'beta': beta_res},
            opt_res.history,
        )
