# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass
from typing import Callable
from numpy.typing import NDArray

import numpy as np

from QHyper.optimizers.util import run_parallel

from .base import Optimizer, OptimizationResult, OptimizerError


@dataclass
class Random(Optimizer):
    number_of_samples: int = 100
    processes: int = 1

    def _minimize(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray
    ) -> OptimizationResult:
        """Returns hyperparameters which lead to the lowest values
            returned by the optimizer

        Parameters
        ----------
        func_creator : Callable[[ArgsType], Callable[[ArgsType], float]]
            function which receives hyperparameters and returns
            a function which will be optimized using the optimizer
        optimizer : Optimizer
            object of the Optimizer class
        init : ArgsType
            initial args for the optimizer
        hyperparams_init : ArgsType
            initial hyperparameters
        evaluation_func : Callable[[ArgsType], Callable[[ArgsType], float]]
            function, which receives hyperparameters, and returns
            function which receives params and return evaluation
        bounds : list[float]
            bounds for hyperparameters (default None)

        Returns
        -------
        ArgsType
            Returns hyperparameters which lead to the lowest values
            returned by the optimizer
        """
        if self.bounds is None:
            raise OptimizerError("This optimizer requires bounds")

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
