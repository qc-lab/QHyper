# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import multiprocessing as mp
from dataclasses import dataclass, field

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from QHyper.optimizers.util import run_parallel

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError)


@dataclass
class CEM(Optimizer):
    epochs: int = 5
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    processes: int = mp.cpu_count()
    n_elite: int = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_elite: int = max(
            int(self.samples_per_epoch * self.elite_frac), 1)

    """Implementation of the Cross Entropy Method for hyperparameter tuning

    Attributes
    ----------
    epochs : int
        number of epochs (default 10)
    samples_per_epoch : int
        number of samples in each epoch (default 100)
    elite_frac : float
        indicate the percent of how many top samples will be used to calculate
        the mean and cov for the next epoch (default 0.1)
    processes : int
        number of processors that will be used (default cpu count)
    n_elite : int
        calulated by multiplying samples_per_epoch by elite_frac
    disable_tqdm: bool
        if set to True, tdqm will be disabled
    verbose: bool
        if set to True, additional information will be printed (default False)
    """

    def _get_points(
        self, mean: NDArray, cov: NDArray
    ) -> NDArray:
        if self.bounds is None:
            raise OptimizerError('This optimizer requires bounds')

        # TODO
        hyperparams: list[NDArray] = []
        while len(hyperparams) < self.samples_per_epoch:
            point = np.random.multivariate_normal(mean, cov)
            if (
                    (self.bounds[:, 0] <= point).all()
                    and (point < self.bounds[:, 1]).all()
            ):
                hyperparams.append(point)

        return np.vstack(hyperparams)

    def _minimize(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray,
    ) -> OptimizationResult:
        """Returns hyperparameters which lead to the lowest values
            returned by optimizer

        Parameters
        ----------
        func_creator : Callable[[ArgsType], Callable[[ArgsType], float]]
            function which receives hyperparameters, and returns
            a function which will be optimized using an optimizer
        optimizer : Optimizer
            object of the Optimizer class
        init : ArgsType
            initial args for optimizer
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
            hyperparameters which lead to the lowest values
            returned by the optimizer
        """

        _init = np.array(init)
        mean = _init.flatten()
        cov = np.identity(len(mean))
        best_hyperparams = _init
        best_result = OptimizationResult(np.inf, _init, [])
        history: list[list[OptimizationResult]] = []

        for i in range(self.epochs):
            if self.verbose:
                print(f'Epoch {i+1}/{self.epochs}')

            hyperparams = self._get_points(mean, cov)
            results = run_parallel(func, hyperparams, self.processes,
                                   self.disable_tqdm)

            elite_ids = np.array(
                [x.value for x in results]).argsort()[:self.n_elite]

            elite_weights = [hyperparams[i].flatten() for i in elite_ids]
            elite_weights = hyperparams[elite_ids]

            if self.verbose:
                print(f'Values: {sorted([x.value for x in results])}')

            if results[elite_ids[0]].value < best_result.value:
                if self.verbose:
                    print(f'New best result: {results[elite_ids[0]].value}')

                best_hyperparams = hyperparams[elite_ids[0]]
                best_result = results[elite_ids[0]]
            history.append([OptimizationResult(res.value, params, [[res]])
                            for res, params in zip(results, hyperparams)])
            mean = np.mean(elite_weights, axis=0)
            cov = np.cov(np.stack((elite_weights), axis=1), bias=True)

        return OptimizationResult(
                best_result.value, best_hyperparams, history)
