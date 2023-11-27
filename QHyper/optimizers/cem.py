import multiprocessing as mp
from dataclasses import dataclass, field

from typing import Callable
import numpy.typing as npt

import numpy as np
import tqdm

from .base import Optimizer, OptimizationResult


@dataclass
class CEM(Optimizer):
    bounds: npt.NDArray[np.float64]
    epochs: int = 5
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    processes: int = mp.cpu_count()
    disable_tqdm: bool = False
    verbose: bool = False
    n_elite: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_elite: int = max(
            int(self.samples_per_epoch * self.elite_frac), 1)
        self.bounds = np.array(self.bounds)

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
        self, mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]
    ) -> list[npt.NDArray[np.float64]]:
        hyperparams: list[npt.NDArray[np.float64]] = []
        while len(hyperparams) < self.samples_per_epoch:
            point = np.random.multivariate_normal(mean, cov)
            if (
                    (self.bounds[:, 0] <= point).all()
                    and (point < self.bounds[:, 1]).all()
            ):
                hyperparams.append(point)

        return hyperparams

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
        init: npt.NDArray[np.float64],
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
        best_result = None
        history: list[list[OptimizationResult]] = []

        for i in range(self.epochs):
            if self.verbose:
                print(f'Epoch {i+1}/{self.epochs}')

            hyperparams = self._get_points(mean, cov)
            with mp.Pool(processes=self.processes) as p:
                results = list(tqdm.tqdm(
                    p.imap(func,
                           [h.reshape(_init.shape) for h in hyperparams]),
                    total=len(hyperparams),
                    disable=self.disable_tqdm,
                ))
            elite_ids = np.array(
                [x.value for x in results]).argsort()[:self.n_elite]

            elite_weights = [hyperparams[i].flatten() for i in elite_ids]

            if self.verbose:
                print(f'Values: {sorted([x.value for x in results])}')

            if (best_result is None
                    or results[elite_ids[0]].value < best_result.value):
                if self.verbose:
                    print(f'New best result: {results[elite_ids[0]].value}')

                best_hyperparams = hyperparams[elite_ids[0]]
                best_result = results[elite_ids[0]]
            history.append([OptimizationResult(res.value, params, [[res]])
                            for res, params in zip(results, hyperparams)])
            mean = np.mean(elite_weights, axis=0)
            cov = np.cov(np.stack((elite_weights), axis=1), bias=True)

        return OptimizationResult(best_result.value, best_hyperparams, history)
