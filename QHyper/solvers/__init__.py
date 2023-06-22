from typing import Type

from .base import Solver

from .vqa.base import VQA
from .gurobi.gurobi import Gurobi
from .cqm.cqm import CQM


SOLVERS: dict[str, Type[Solver]] = {
    'vqa': VQA,
    'gurobi': Gurobi,
    'cqm': CQM,
}
