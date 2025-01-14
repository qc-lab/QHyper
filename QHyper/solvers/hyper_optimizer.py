from dataclasses import dataclass, field

from numpy.typing import NDArray
from typing import Any
from QHyper.optimizers import OptimizationResult, Optimizer, OptimizationParameter
from QHyper.solvers import Solver, SolverResult
from QHyper.problems import Problem
from QHyper.util import weighted_avg_evaluation


class HyperOptimizerProperty:
    def __init__(self, name: str, min_value: list[float],
                 max_value: list[float], initial_value: list[float]) -> None:
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.initial_value = initial_value

        if len(min_value) != len(max_value):
            raise ValueError(
                "min_value and max_value must have the same length")
        if len(min_value) != len(initial_value):
            raise ValueError(
                "min_value and initial_value must have the same length")

    def get_bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.min_value, self.max_value))


@dataclass
class HyperOptimizer:
    """ Class for hyperparameter optimization

    HyperOptimizer is a class that allows to use the optimizers and 
    find the best parameters for a given solver.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer to use for optimization
    solver : Solver
        The solver to use for optimization
    properties : dict[str, OptimizationParameter]
        The properties to optimize. Their keys must match the properties 
        of the solver. 
    history : list[SolverResult]
        The history of the optimization process
    best_params : OptimizationResult
        The best parameters found during optimization
    """

    optimizer: Optimizer
    solver: Solver
    properties: dict[str, OptimizationParameter] = field(default_factory=dict)
    history: list[SolverResult] = field(default_factory=list)
    best_params: OptimizationResult = field(init=False)

    def __init__(self, optimizer: Optimizer, solver: Solver, **properties: Any) -> None:
        self.optimizer = optimizer
        self.solver = solver
        self.properties = {}
        self.history = []

        for property, values in properties.items():
            self.properties[property] = OptimizationParameter(**values)

    @property
    def problem(self) -> Problem:
        return self.solver.problem

    def parse_params(self, params: NDArray) -> dict[str, list[float]]:
        parsed_params = {}

        param_index = 0
        for property, opt_param in self.properties.items():
            parsed_params[property] = params[
                param_index:param_index + len(opt_param)]
            param_index += len(opt_param)
        return parsed_params

    def run_solver(self, params: NDArray) -> SolverResult:
        return self.solver.solve(**self.parse_params(params))

    def _optimization_function(self, params: NDArray) -> OptimizationResult:
        result = self.run_solver(params)

        value = weighted_avg_evaluation(
            result.probabilities, self.solver.problem.get_score,
        )
        self.history.append(result)

        return OptimizationResult(
            value=value,
            params=params,
            history=[]
        )

    def solve(self) -> OptimizationResult:
        initial_params = sum(self.properties.values(), OptimizationParameter())

        self.best_params = self.optimizer.minimize(
            self._optimization_function, initial_params)
        return self.best_params

    def run_with_best_params(self) -> SolverResult:
        if self.best_params is None:
            raise ValueError(
                "Run hyper optimization first. Call solve() method")
        return self.run_solver(self.best_params.params)
