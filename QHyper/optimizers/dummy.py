from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .base import Optimizer, OptimizationResult


@dataclass
class Dummy(Optimizer):
    def minimize(
            self,
            func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
            init: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        return func(init)
