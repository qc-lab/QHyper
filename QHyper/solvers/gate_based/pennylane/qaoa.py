import pennylane as qml
from pennylane import numpy as np

from numpy.typing import NDArray
from typing import Any, Callable, cast, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult, Optimizer, Dummy

from QHyper.converter import Converter
from QHyper.polynomial import Polynomial
from QHyper.solvers.base import Solver, SolverResult, SolverException

# from .mixers import MIXERS_BY_NAME


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
    angles: NDArray
    hyper_args: NDArray
    qubo_cache: dict[tuple[float, ...], qml.Hamiltonian] = {}
    dev: qml.Device | None = None

    def __init__(
            self, problem: Problem,
            angles: NDArray, hyper_args: NDArray,
            optimizer: Optimizer = Dummy(),
            layers: int = 3, backend: str = "default.qubit",
            mixer: str = "pl_x_mixer"
    ) -> None:
        self.problem = problem
        self.optimizer = optimizer
        self.angles = angles
        self.hyper_args = hyper_args
        self.layers = layers
        self.backend = backend
        self.mixer = mixer

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
        if self.mixer not in MIXERS_BY_NAME:
            raise Exception(f"Unknown {self.mixer} mixer")
        return MIXERS_BY_NAME[self.mixer](
            [str(v) for v in cost_operator.wires])

    def _circuit(
        self,
        params: NDArray,
        cost_operator: qml.Hamiltonian,
    ) -> None:
        def qaoa_layer(gamma: NDArray, beta: NDArray) -> None:
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(
                beta, self._create_mixing_hamiltonian(cost_operator))

        self._hadamard_layer(cost_operator)
        params = params.reshape(2, -1)
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def get_expval_circuit(
        self, weights: list[float]
    ) -> Callable[[NDArray], OptimizationResult]:
        cost_operator = self.create_cost_operator(self.problem, weights)

        self.dev = qml.device(self.backend, wires=cost_operator.wires)

        @qml.qnode(self.dev)
        def expval_circuit(params: NDArray) -> OptimizationResult:
            self._circuit(params, cost_operator)
            return OptimizationResult(
                    cast(float, qml.expval(cost_operator)), params)

        return expval_circuit

    def get_probs_func(
        self, problem: Problem, weights: list[float]
    ) -> Callable[[NDArray], NDArray]:
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
        def probability_circuit(params: NDArray) -> NDArray:
            self._circuit(params, cost_operator)
            return cast(
                NDArray, qml.probs(wires=cost_operator.wires)
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
        angles: NDArray,
        weights: list[float],
    ) -> np.recarray:
        probs = self.get_probs_func(problem, weights)(
            angles.reshape(2, -1))

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

    def get_params_init_format(
        self, opt_args: NDArray, hyper_args: NDArray
    ) -> dict[str, Any]:
        return {
            "angles": opt_args,
            "hyper_args": hyper_args,
        }

    def solve(self, angles: NDArray | None, weights: list[float] | None) -> SolverResult:
        if not angles and not self.angles:
            raise SolverException("Parameter 'angles' was not provided")
        angles = angles if angles else self.angles

        if not weights:
            weights = [1.] * (len(self.problem.constraints) + 1)

        # opt_wrapper = LocalOptimizerFunction(
        #         self.pqc, self.problem, best_hargs)
        # opt_res = self.optimizer.minimize(opt_wrapper, opt_args)

        opt_res = self.optimizer.minimize(
            self.get_expval_circuit(weights), angles)

        return SolverResult(
            self.run_with_probs(self.problem, opt_res.params, weights),
            {'angles': opt_res.params, 'weights': weights},
            opt_res.history,
        )
