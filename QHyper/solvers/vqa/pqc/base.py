from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from typing import Any, Optional

from QHyper.problems.base import Problem


PQCResults = tuple[dict[str, float], list[float]]


class PQC:
    pqc_type: str

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def run(
        self, 
        problem: Problem, 
        args: npt.NDArray[np.float64], 
        hyper_args: npt.NDArray[np.float64]
    ) -> PQCResults: ...

    @abstractmethod
    def get_params(
        self, 
        params_inits: dict[str, Any], 
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
