from .base import EvalFunc
from QHyper.problems.base import Problem


class WFEval(EvalFunc):
    def __init__(self, penalty: float = 0) -> None:
        self.penalty = penalty

    def evaluate(
            self,
            results: dict[str, float],
            problem: Problem,
            const_params: list[float]
    ) -> float:
        score: float = 0

        sorted_results = {
            k: v for k, v in
            sorted(
                results.items(),
                key=lambda item: item[1],
                reverse=True
            )[:40]
        }

        # print(sorted_results)
        scaler = 1/sum([v for v in sorted_results.values()])
        # print(scaler)

        for result, prob in sorted_results.items():
            score += scaler * prob * problem.get_score(result, self.penalty)
        return score
