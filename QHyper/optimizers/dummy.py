# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray

from QHyper.optimizers.base import Optimizer, OptimizationResult


@dataclass
class Dummy(Optimizer):
    def _minimize(
            self,
            func: Callable[[NDArray], OptimizationResult],
            init: NDArray
    ) -> OptimizationResult:
        result = func(init)
        return OptimizationResult(result.value, result.params, [[result]])
