from abc import abstractmethod

from QHyper.problems.base import Problem


class EvalFunc:
    @abstractmethod
    def evaluate(
        self,
        results: dict[str, float],
        problem: Problem,
        const_params: list[float]
    ) -> float: ...
