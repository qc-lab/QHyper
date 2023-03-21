from .base import EvalFunc
from QHyper.problems.base import Problem
from QHyper.solvers.vqa.pqc.base import PQCResults

class ExpVal(EvalFunc):
    def evaluate(self, results: PQCResults, problem: Problem, const_params: list[float]) -> float: 
        ...
