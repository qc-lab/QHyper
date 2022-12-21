from ..solver import Solver
from .pennylane import PennyLaneQAOA


def QAOA(**kwargs) -> Solver:
    """
    Parameters
    ----------
    kwargs : dict[str, Any]
        argumets that will be passed to Solver contructor. 
        Requires `platform` argument (currently only "pennylane" is supported)

    Returns
    -------
    Solver
        Returns solver object of class chose by `platform` argument
    
    """

    platform = kwargs.pop('platform')

    if platform == "pennylane":
        return PennyLaneQAOA(**kwargs)

    raise Exception(f"Unknown platform {platform}")
