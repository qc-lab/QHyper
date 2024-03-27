# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


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
        result = func(init)
        return OptimizationResult(result.value, result.params, [[result]])
