# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable
import numpy as np
from numpy.typing import NDArray

from QHyper.optimizers.util import run_parallel
from QHyper.optimizers.base import (
    OptimizationResult, Optimizer, OptimizerError)


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

    bounds: NDArray
    steps: list[float]
    verbose: bool
    disable_tqdm: bool
    processes: int = 1

    def __init__(
        self,
        bounds: NDArray,
        steps: list[float],
        verbose: bool = False,
        disable_tqdm: bool = True,
        processes: int = 1,
    ) -> None:
        self.bounds = bounds
        self.steps = steps
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.processes = processes

    def _generate_grid(self) -> NDArray:
        if self.bounds is None:
            raise OptimizerError("This optimizer requires bounds")

        return np.stack(
            np.meshgrid(
                *[np.arange(bound[0], bound[1], step)
                  for bound, step in zip(self.bounds, self.steps)]
            ), axis=-1
        ).reshape(-1, len(self.bounds))

    def minimize_(
            self,
            func: Callable[[NDArray], OptimizationResult],
            init: NDArray | None
    ) -> OptimizationResult:
        self.check_bounds(None)

        hyperparams = self._generate_grid()

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
