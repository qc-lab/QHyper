# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable
from numpy.typing import NDArray

import numpy as np

from QHyper.optimizers.util import run_parallel

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError)


class Random(Optimizer):
    """Random optimizer

    Attributes
    ----------
    bounds : numpy.ndarray
        The bounds for the optimization algorithm. Not all optimizers
        support bounds. The shape of the array should be (n, 2), where
        n is the number of parameters (`init` in method :meth:`minimize`).
    verbose : bool, default False
        Whether to print the optimization progress.
    disable_tqdm : bool, default True
        Whether to disable the tqdm progress bar.
    number_of_samples : int, default 100
        The number of samples to generate.
    processes : int, default 1
        The number of processes to use for parallel computation.
    """

    bounds: NDArray
    verbose: bool
    disable_tqdm: bool
    number_of_samples: int = 100
    processes: int = 1

    def __init__(
        self,
        bounds: NDArray,
        verbose: bool = False,
        disable_tqdm: bool = True,
        number_of_samples: int = 100,
        processes: int = 1,
    ) -> None:
        self.bounds = bounds
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.number_of_samples = number_of_samples
        self.processes = processes

    def minimize_(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray | None
    ) -> OptimizationResult:
        if init is None:
            raise OptimizerError("Initial point must be provided.")
        self.check_bounds(init)

        hyperparams = (
            (self.bounds[:, 1] - self.bounds[:, 0])
            * np.random.rand(self.number_of_samples, *init.shape)
            + self.bounds[:, 0])

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
            history=[history],
        )
