from .base import EvalFunc
from QHyper.problems.base import Problem
from QHyper.solvers.vqa.pqc.base import PQCResults


class WFEval(EvalFunc):
    def evaluate(self, results: PQCResults, problem: Problem, const_params: list[float]) -> float: 
        score = 0
        for result, prob in results.items():
            if (value := problem.get_score(result)) is None:
                score += 0
            else:
                score -= prob * value
        return score
