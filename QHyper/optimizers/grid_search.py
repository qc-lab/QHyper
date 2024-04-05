# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass

from typing import Callable
import numpy as np
from numpy.typing import NDArray

from QHyper.optimizers.util import run_parallel
from .base import OptimizationResult, Optimizer, OptimizerError


@dataclass
class GridSearch(Optimizer):
    """
    Parameters
    ----------
    bounds : list[tuple[float, float]]
        list of tuples with lower and upper bounds for each variable
    steps : list[float]
        step for each variable bound
    processes : int
        number of processors that will be used (default cpu count)
    disable_tqdm: bool
        if set to True, tdqm will be disabled (default False)
    verbose: bool
        if set to True, additional information will be printed
        (default False)
    """
    steps: list[float]
    processes: int = 1

    def generate_grid(self) -> NDArray:
        """
        Generates grid of hyperparameters based on bounds and steps

        Returns
        -------
        numpy.ndarray
            grid of hyperparameters
        """
        if self.bounds is None:
            raise OptimizerError("This optimizer requires bounds")

        return np.stack(
            np.meshgrid(
                *[np.arange(bound[0], bound[1], step)
                  for bound, step in zip(self.bounds, self.steps)]
            ), axis=-1
        ).reshape(-1, len(self.bounds))

    def _minimize(
            self,
            func: Callable[[NDArray], OptimizationResult],
            init: NDArray
    ) -> OptimizationResult:
        if self.bounds is None:
            raise OptimizerError("This optimizer requires bounds")

        hyperparams = self.generate_grid()

        results = run_parallel(func, hyperparams, self.processes, self.disable_tqdm)
        min_idx = np.argmin([result.value for result in results])

        if self.verbose:
            print(f"Best result: {results[min_idx].value}")
            print(f"Values: {sorted([v.value for v in results])}")

        history = [OptimizationResult(res.value, params, [[res]])
                   for res, params in zip(results, hyperparams)]
        return OptimizationResult(
            value=results[min_idx].value,
            params=hyperparams[min_idx],
            history=[history]
        )
