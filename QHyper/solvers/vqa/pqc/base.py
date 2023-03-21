from typing import Any

from QHyper.problems.base import Problem


PQCResults = dict[str, float]


class PQC:
    pqc_type: str
    def __init__(self, **kwargs) -> None: ...

    def run(self, problem: Problem, args: list[float], hyper_args: list[float]) -> PQCResults: ...

    def get_params(self, params_inits: dict[str, Any], hyper_args: list[float] = []) -> tuple[list[float], list[float]]: ...
