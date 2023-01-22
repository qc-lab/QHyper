import numpy as np
import pennylane as qml

from ...optimizers.optimizer import HyperparametersOptimizer, Optimizer
from ...problems.problem import Problem
from ..solver import Solver
from .parser import parse_hamiltonian


class PennyLaneQAOA(Solver):
    """QAOA solver based on PennyLane

    Attributes
    ----------
    problem : Problem
        problem definition
    angles: list[float]
        angles for the QAOA, also initial values for optimizers
    optimizer : Optmizer
        optimizer that will be used to find proper QAOA angles (default None)
    layers : int
        number of QAOA layers in the circuit (default 3)
    mixer : str
        mixer name (currently only "X" is supported) (default "X")
    weights : list[float]
        needed for converting Problem to QUBO, if QUBO is already provided
        weights should be [1] (default [1])
    hyperoptimizer : HyperparametersOptimizer
            optimizer to tune the weights (default None)
    backend : str
        name of the backend (default "default.qubit")
    dev : qml.device
        PennyLane quantum device
    """

    def __init__(self, **kwargs) -> None:
        """
        Parameters
        ----------
        problem : Problem
            problem definition
        angles: list[float]
            angles for QAOA, also initial values for optimizers
        optimizer : Optmizer
            optimizer that will be used to find proper QAOA angles (default None)
        layers : int
            number of QAOA layers in the circuit (default 3)
        mixer : str
            mixer name (currently only "X" is supported) (default "X")
        weights : list[float]
            needed for converting Problem to QUBO, if QUBO is already provided
            weights should be [1] (default [1])
        hyperoptimizer : HyperparametersOptimizer
            optimizer to tune the weights (default None)
        backend : str
            name of the backend (default "default.qubit")
        """

        self.problem: Problem = kwargs.get("problem")
        self.angles: list[float] = kwargs.get("angles")

        self.optimizer: Optimizer = kwargs.get("optimizer", None)
        self.layers: int = kwargs.get("layers", 3)
        self.mixer: str = kwargs.get("mixer", "X")
        self.weights: list[float] = kwargs.get("weights", [1])
        self.hyperoptimizer: HyperparametersOptimizer = kwargs.get("hyperoptimizer", None)
        self.backend: str = kwargs.get("backend", "default.qubit")

        self.dev = qml.device(self.backend, wires=self.problem.variables)

    def _create_cost_operator(self, weights: list[float]) -> qml.Hamiltonian:
        elements = [self.problem.objective_function] + self.problem.constraints
        cost_operator = ""
        if len(weights) != len(elements):
            raise Exception(
                f"Number of provided weights ({len(weights)}) is different from number of elements ({len(elements)})")
        for weight, ingredient in zip(weights, elements):
            cost_operator += f"+{weight}*({ingredient})"
        return parse_hamiltonian(cost_operator)

    def _hadamard_layer(self):
        for i in range(self.problem.variables):
            qml.Hadamard(i)

    def _create_mixing_hamiltonian(self) -> qml.Hamiltonian:
        if self.mixer == "X":
            return qml.qaoa.x_mixer(range(self.problem.variables))
        # REQUIRES GRAPH https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.mixers.xy_mixer.html
        # if self.mixer == "XY": 
        #     return qml.qaoa.xy_mixer(...)
        raise Exception(f"Unknown {self.mixer} mixer")

    def _circuit(self, params: tuple[list[float], list[float]], cost_operator: qml.Hamiltonian):
        def qaoa_layer(gamma, beta):
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(beta, self._create_mixing_hamiltonian())

        self._hadamard_layer()
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def get_expval_func(self, weights: list[float]):
        """Take angles and return the expectation value

        Parameters
        ----------
        weights : list[float]
            weights for converting Problem to QUBO
        
        Returns
        -------
        Callable[[list[float]], float]
            Returns function that takes angles and returns expectation value
        """

        cost_operator = self._create_cost_operator(weights)

        @qml.qnode(self.dev)
        def cost_function(params):
            self._circuit(params, cost_operator)
            x = qml.expval(cost_operator)
            return x

        return cost_function

    def get_probs_val_func(self, weights):
        """Returns function that takes angles and returns score

        Parameters
        ----------
        weights : list[float]
            weights for converting Problem to QUBO
        
        Returns
        -------
        Callable[[list[float]], float]
            Returns function that takes angles and returns score of the probabilities based
            on the user defined function `problem.get_score()`
        """

        def probability_value(params):
            probs = self.get_probs_func(weights)(params)
            return self.check_results(probs)

        return probability_value
    
    def evaluate(self, weights, params):
        """Returns evaluation of given parameters 

        Parameters
        ----------
        weights : list[float]
            weights for converting Problem to QUBO
        params : list[float]
            angles for QAQA Problem
        
        Returns
        -------
        float
            Returns evaluation of given parameters
        """

        probs = self.get_probs_func(weights)(params)
        return self.check_results(probs)


    def get_probs_func(self, weights):
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

        cost_operator = self._create_cost_operator(weights)

        @qml.qnode(self.dev)
        def probability_circuit(params):
            self._circuit(params, cost_operator)
            return qml.probs(wires=range(self.problem.variables))

        return probability_circuit

    def check_results(self, probs: dict[str, float]) -> float:
        """Returns score based on user defined function `problem.get_score()`

        Parameters
        ----------
        probs : dict[str, float]
            probabilities of each state
        
        Returns
        -------
        float
            Score based on user defined function `problem.get_score()`
            Adds 0 if state is not correct (`problem.get_score()` returned None), 
            or subtract probability * `problem.get_score()` from the final result.
            The smaller the returned value, the better the probabilities.
        """

        to_bin = lambda x: format(x, 'b').zfill(self.problem.variables)

        results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
        results_by_probabilites = dict(
            sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
        score = 0
        for result, prob in results_by_probabilites.items():
            if (value := self.problem.get_score(to_bin(result))) is None:
                score += 0
            else:
                score -= prob * value
        return score

    def print_results(self, probs: dict[str, float]) -> None:
        """Prints to std output probabilities of each state and its score
        based on user defined function `problem.get_score()`.

        Parameters
        ----------
        probs : dict[str, float]
            probabilities of each state
        """

        to_bin = lambda x: format(x, 'b').zfill(self.problem.variables)
        results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
        results_by_probabilites = dict(
            sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
        for result, prob in results_by_probabilites.items():
            # binary_rep = to_bin(key)
            value = self.problem.get_score(to_bin(result))
            print(
                f"Key: {to_bin(result)} with probability {prob:.5f}   "
                f"| correct: {'True, value: ' + format(value, '.5f') if value is not None else 'False'}"
            )

    def solve(self) -> tuple[float, list[float], list[float]]:
        """Run optimizer and hyperoptimizer (if provided)
        If hyperoptimizer is provided in constructor, weights will be optimized first.
        Then optimizer takes these weights and returns angles which give the best probabilities.

        Returns
        -------
        tuple[float, list[float], list[float]]
            Returns tuple of score, angles, weights
        """

        if self.hyperoptimizer:
            return self.hyperoptimizer.minimize(
            func_creator=self.get_expval_func, 
            optimizer=self.optimizer, 
            init=np.array(self.angles), 
            hyperparams_init=np.array(self.weights), 
            evaluation_func=self.evaluate,
            # bounds=[0.001, 10] #TODO
        )
        if self.optimizer:
            params, _ = self.optimizer.minimize(self.get_expval_func(self.weights), self.angles)
        else:
            params = self.angles
        return self.evaluate(self.weights, params), params, self.weights
