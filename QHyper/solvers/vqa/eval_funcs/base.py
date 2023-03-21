from QHyper.problems.base import Problem
from QHyper.solvers.vqa.pqc.base import PQCResults


class EvalFunc:
    def evaluate(self, results: PQCResults, problem: Problem, const_params: list[float]) -> float: ...
