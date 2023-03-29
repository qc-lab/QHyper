from .base import EvalFunc
from QHyper.problems.base import Problem


class ExpVal(EvalFunc):
    def evaluate(self, results: dict[str, float], problem: Problem, const_params: list[float]) -> float: 
        raise NotImplemented
