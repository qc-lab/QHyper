# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from .base import EvalFunc
from QHyper.problems.base import Problem


class WFEval(EvalFunc):
    def __init__(self, penalty: float = 0,
                 limit_results: int | None = None) -> None:
        self.penalty = penalty
        self.limit_results = limit_results

    def evaluate(
            self,
            results: dict[str, float],
            problem: Problem,
            const_params: list[float],
    ) -> float:
        score: float = 0

        limit_results = self.limit_results or len(results)

        sorted_results = {
            k: v for k, v in
            sorted(
                results.items(),
                key=lambda item: item[1],
                reverse=True
            )[:limit_results]
        }

        scaler = 1/sum([v for v in sorted_results.values()])

        for result, prob in sorted_results.items():
            score += scaler * prob * problem.get_score(result, self.penalty)
        return score
