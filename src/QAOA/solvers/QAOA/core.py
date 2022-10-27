
from .Pennylane import PennyLaneQAOA
from ..solver import Solver


def QAOA(**kwargs) -> Solver:
    platform = kwargs.pop('platform')

    if platform == "pennylane":
        return PennyLaneQAOA(**kwargs)

    raise Exception(f"Unknown platform {platform}")
