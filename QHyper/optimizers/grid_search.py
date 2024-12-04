# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable
import numpy as np
from numpy.typing import NDArray

from QHyper.optimizers.util import run_parallel
from QHyper.optimizers.base import (
    OptimizationResult, Optimizer, OptimizerError, OptimizationParameter)


class GridSearch(Optimizer):
    """
    Grid search optimizer

    Attributes
    ----------
    bounds : numpy.ndarray, optional
        The bounds for the optimization algorithm. Not all optimizers
        support bounds. The shape of the array should be (n, 2), where
        n is the number of parameters (`init` in method :meth:`minimize`).
    steps : list[float]
        The step for each bound. The length of the list should be equal
        to the number of bounds. E.g. for bounds [[0, 1]]
        the steps could be [0.1].
        The searching space will be: [0, 0.1, 0.2, ..., 0.9]
    verbose : bool, default False
        Whether to print the optimization progress.
    disable_tqdm : bool, default True
        Whether to disable the tqdm progress bar.
    processes : int, default 1
        The number of processes to use for parallel computation.
    """

    # bounds: NDArray
    verbose: bool
    disable_tqdm: bool
    processes: int = 1

    def __init__(
        self,
        # bounds: NDArray,
        verbose: bool = False,
        disable_tqdm: bool = True,
        processes: int = 1,
    ) -> None:
        # self.bounds = bounds
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.processes = processes

    def _generate_grid(self, params: OptimizationParameter
                       ) -> list[list[float]]:
        return np.stack(
            np.meshgrid(
                *[np.arange(min_, max_, step)
                  for min_, max_, step
                  in zip(params.min, params.max, params.step)]
            ), axis=-1
        ).reshape(-1, len(params))

    def minimize_(
            self,
            func: Callable[[NDArray], OptimizationResult],
            init: OptimizationParameter | None
    ) -> OptimizationResult:
        if init is None:
            raise OptimizerError("Optimization parameter must be provided.")
        init.assert_bounds()
        init.assert_step()

        hyperparams = self._generate_grid(init)
        results = run_parallel(
            func, hyperparams, self.processes, self.disable_tqdm)
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
