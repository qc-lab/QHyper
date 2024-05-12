# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable

import numpy as np
from numpy.typing import NDArray

from QHyper.optimizers.util import run_parallel

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError)


class CEM(Optimizer):
    """Implementation of the Cross Entropy Method for hyperparameter tuning

    Attributes
    ----------
    bounds : numpy.ndarray, optional
        The bounds for the optimization algorithm. Not all optimizers
        support bounds. The shape of the array should be (n, 2), where
        n is the number of parameters (`init` in method :meth:`minimize`).
    verbose : bool, default False
        Whether to print the optimization progress.
    disable_tqdm : bool, default True
        Whether to disable the tqdm progress bar.
    epochs : int, default 5
        The number of epochs.
    samples_per_epoch : int, default 100
        The number of samples per epoch.
    elite_frac : float, default 0.1
        The fraction of elite samples that will be used to update the
        mean and covariance for next epoch.
    processes : int, default 1
        The number of processes to use for parallel computation.
    n_elite : int
        The number of elite samples. Calculated as
        `samples_per_epoch * elite_frac`.
    """

    bounds: NDArray
    verbose: bool
    disable_tqdm: bool
    epochs: int
    samples_per_epoch: int
    elite_frac: float
    processes: int
    n_elite: int

    def __init__(
        self,
        bounds: NDArray,
        verbose: bool = False,
        disable_tqdm: bool = True,
        epochs: int = 5,
        samples_per_epoch: int = 100,
        elite_frac: float = 0.1,
        processes: int = 1,
    ) -> None:
        """
        Parameters
        ----------

        bounds : numpy.ndarray
        verbose : bool, default False
        disable_tqdm : bool, default True
        epochs : int, default 5
        samples_per_epoch : int, default 100
        elite_frac : float, default 0.1
        processes : int, default 1
        """

        self.bounds = bounds
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.epochs = epochs
        self.samples_per_epoch = samples_per_epoch
        self.elite_frac = elite_frac
        self.processes = processes

        self.n_elite: int = max(
            int(self.samples_per_epoch * self.elite_frac), 1)

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

    def minimize_(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray | None,
    ) -> OptimizationResult:
        if init is None:
            raise OptimizerError("Initial point must be provided.")

        self.check_bounds(init)

        mean = init.flatten()
        cov = np.identity(len(mean))
        best_hyperparams = init
        best_result = OptimizationResult(np.inf, init, [])
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
