# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError, OptimizationParameter)


class Dummy(Optimizer):
    """
    Dummy optimizer.

    This optimizer is used as a default optimizer in the case
    when no optimizer is selected. It simply evaluates the function.
    It requires the initial point to be provided.

    """

    def minimize(
        self,
        func: Callable[[list[float]], OptimizationResult],
        init: OptimizationParameter | None,
    ) -> OptimizationResult:
        if init is None:
            raise OptimizerError("Initial point must be provided.")
        init.assert_init()

        result = func(init.init)
        return OptimizationResult(result.value, result.params, [[result]])
