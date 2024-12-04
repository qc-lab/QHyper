from dataclasses import dataclass, field

from numpy.typing import NDArray
from typing import Any
from QHyper.optimizers import OptimizationResult, Optimizer, OptimizationParameter
from QHyper.solvers import Solver, SolverResult
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
class HyperOptimizer(Solver):
    optimizer: Optimizer
    solver: Solver
    properties: dict[str, OptimizationParameter] = field(default_factory=dict)
    history: list[SolverResult] = field(default_factory=list)

    def __init__(self, optimizer: Optimizer, solver: Solver, **properties: Any) -> None:
        self.optimizer = optimizer
        self.solver = solver
        self.properties = {}
        self.history = []

        for property, values in properties.items():
            self.properties[property] = OptimizationParameter(**values)

    def optimization_function(self, params: NDArray) -> OptimizationResult:
        print(params)
        solver_params = {}

        param_index = 0
        for property, opt_param in self.properties.items():
            solver_params[property] = params[
                    param_index:param_index + len(opt_param)]
            param_index += len(opt_param)
        # print(solver_params)
        result = self.solver.solve(**solver_params)

        value = weighted_avg_evaluation(
            result.probabilities, self.solver.problem.get_score,
        )
        self.history.append(result)

        return OptimizationResult(
            value=value,
            params=params,
            history=[]
        )

    def solve(self) -> SolverResult:
        initial_params = sum(self.properties.values(), OptimizationParameter())

        result = self.optimizer.minimize(
            self.optimization_function, initial_params)
        return result
