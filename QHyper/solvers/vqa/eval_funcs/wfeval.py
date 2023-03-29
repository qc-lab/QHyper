from .base import EvalFunc
from QHyper.problems.base import Problem


class WFEval(EvalFunc):
    def evaluate(self, results: dict[str, float], problem: Problem, const_params: list[float]) -> float: 
        score: float = 0
        for result, prob in results.items():
            if (value := problem.get_score(result)) is None:
                score += 0
            else:
                score -= prob * value
        return score
