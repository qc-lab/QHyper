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
    def run_opt(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> float: ...

    @abstractmethod
    def get_opt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        ...

    @abstractmethod
    def get_hopt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        ...

    @abstractmethod
    def get_init_args(
        self,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        ...
