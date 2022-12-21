from ..solver import Solver
from ...problems.problem import Problem


class CQM(Solver):
    def __init__(self, **kwargs) -> None:
        self.problem: Problem = kwargs.get("problem")
        self.time: float = kwargs.get("time", None)
