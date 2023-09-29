from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from .base import Optimizer, OptimizationResult


@dataclass
class Dummy(Optimizer):
    def minimize(
            self, 
            func: Callable[[npt.NDArray[npt.float64]], float], 
            init: npt.NDArray[npt.float64]
    ) -> OptimizationResult:
        return OptimizationResult(func(init), init)
