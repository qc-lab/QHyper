from .base import EvalFunc
from QHyper.problems.base import Problem

from QHyper.hyperparameter_gen.parser import Expression

class ExpVal(EvalFunc):
    def evaluate(self, results: dict[str, float], problem: Problem, const_params: list[float]) -> float:
        problem_function = Expression(
            problem.objective_function.polynomial * const_params[0]).polynomial
        for weight, element in zip(const_params[1:], problem.constraints):
            problem_function += Expression(element.polynomial**2 * weight).polynomial
        exp_val = 0.0
        for solution, probability in results.items():
            result = problem_function.evalf(subs={
                x: y for x, y in zip(problem.variables, solution)
            })
            exp_val += probability * result

        return exp_val