# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable
from numpy.typing import NDArray

import numpy as np

from QHyper.optimizers.util import run_parallel

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError, OptimizationParameter)


class Random(Optimizer):
    """Random optimizer

    The random optimizer is a simple optimization algorithm that
    generates random samples from the parameter space and evaluates
    the function at each point.
    This alogrithm requries the following parameters to be set:
    - `min` and `max` bounds for each parameter

    Attributes
    ----------
    verbose : bool, default False
        Whether to print the optimization progress.
    disable_tqdm : bool, default True
        Whether to disable the tqdm progress bar.
    number_of_samples : int, default 100
        The number of samples to generate.
    processes : int, default 1
        The number of processes to use for parallel computation.
    """

    verbose: bool
    disable_tqdm: bool
    number_of_samples: int = 100
    processes: int = 1

    def __init__(
        self,
        verbose: bool = False,
        disable_tqdm: bool = True,
        number_of_samples: int = 100,
        processes: int = 1,
    ) -> None:
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.number_of_samples = number_of_samples
        self.processes = processes

    def minimize(
        self,
        func: Callable[[list[float]], OptimizationResult],
        init: OptimizationParameter
    ) -> OptimizationResult:
        init.assert_bounds()
        bounds = np.array(init.bounds)

        hyperparams = (
            (bounds[:, 1] - bounds[:, 0])
            * np.random.rand(self.number_of_samples, len(bounds))
            + bounds[:, 0])

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
