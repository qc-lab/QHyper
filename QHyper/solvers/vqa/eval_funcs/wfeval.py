from .base import EvalFunc
from QHyper.problems.base import Problem


class WFEval(EvalFunc):
    def __init__(self, penalty: float = 0) -> None:
        self.penalty = penalty

    def evaluate(self, results: dict[str, float], problem: Problem, const_params: list[float]) -> float: 
        score: float = 0
        for result, prob in results.items():
            if (value := problem.get_score(result)) is None:
                score += prob * self.penalty
            else:
                score -= prob * value
        return score
