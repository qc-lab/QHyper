import multiprocessing as mp
from dataclasses import dataclass, field

from typing import Any, Callable

import numpy as np
import tqdm

from .base import Optimizer


@dataclass
class CEM(Optimizer):
    epochs: int = 10
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    processes: int = mp.cpu_count(),
    print_on_epochs: list[int] = [],
    disable_tqdm: bool = False,
    bounds: list[tuple[float, float]] = None
    n_elite: int = field(init=False)

    def __post_init__(self):
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
    print_on_epochs: list[int]
        list indicating after which epochs print best results
    disable_tqdm: bool
        if set to True, tdqm will be disabled
    """

    # def __init__(
    #     self, epochs: int = 10,
    #     samples_per_epoch: int = 100,
    #     elite_frac: float = 0.1,
    #     processes: int = mp.cpu_count(),
    #     print_on_epochs: list[int] = [],
    #     disable_tqdm: bool = False,
    #     bounds: list[tuple[float, float]] = None
    # ) -> None:
    #     """
    #     Parameters
    #     ----------
    #     epochs : int
    #         number of epochs (default 10)
    #     samples_per_epoch : int
    #         number of samples in each epoch (default 100)
    #     elite_frac : float
    #         indicate the percent of how many top samples will be used to calculate
    #         the mean and cov for the next epoch (default 0.1)
    #     processes : int
    #         number of processors that will be used (default cpu count)
    #     print_on_epochs: list[int]
    #         list indicating after which epochs print best results (default [])
    #     disable_tqdm: bool
    #         if set to True, tdqm will be disabled (default False)
    #     """

    #     self.epochs: int = epochs
    #     self.samples_per_epoch: int = samples_per_epoch
    #     self.elite_frac: float = elite_frac
    #     self.processes: int = processes
    #     self.print_on_epochs: list[int] = print_on_epochs
    #     self.disable_tqdm = disable_tqdm
    #     self.bounds = np.array(bounds)

    def minimize(
        self,
        func: Callable[[list[float]], float],
        init: list[float]
    ) -> tuple[float, list[float]]:
        """Returns hyperparameters which lead to the lowest values returned by optimizer
        
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
            hyperparameters which lead to the lowest values returned by the optimizer
        """
        # wrapper = Wrapper(func_creator, optimizer, evaluation_func, init)
        init = np.array(init)
        mean = init
        cov = np.identity(len(init))
        best_hyperparams = init
        best_score = func(init)

        for i_iteration in range(1, self.epochs+1):
            hyperparams = []
            while len(hyperparams) < self.samples_per_epoch:
                point = np.random.multivariate_normal(mean, cov)
                if (self.bounds[:, 0] <= point).all() and (point < self.bounds[:, 1]).all(): 
                    hyperparams.append(point)
            # hyperparams = np.random.multivariate_normal(mean, cov, size=self.samples_per_epoch)
            # if bounds:
            #     hyperparams[hyperparams < bounds[0]] = bounds[0]
            #     hyperparams[hyperparams > bounds[1]] = bounds[1]

            # hyperparams = np.concatenate((hyperparams, [best_hyperparams.flatten()]), axis=0)
            hyperparams = [
                np.reshape(np.array(hyperparam), init.shape) for hyperparam in hyperparams]

            with mp.Pool(processes=self.processes) as p:
                results = list(tqdm.tqdm(
                    p.imap(func, hyperparams), total=len(hyperparams), disable=self.disable_tqdm))

            elite_idxs = np.array(results).argsort()[:self.n_elite]

            elite_weights = [hyperparams[i].flatten() for i in elite_idxs]

            best_hyperparams = elite_weights[0].reshape(init.shape)

            reward = func(best_hyperparams)
            if reward < best_score:
                best_hyperparams = best_hyperparams
                best_score = reward

            mean = np.mean(elite_weights, axis=0)
            cov = np.cov(np.stack((elite_weights), axis=1), bias=True)

            if i_iteration in self.print_on_epochs:
                # print(f'Epoch {i_iteration}\t'
                #       f'Average Elite Score: {np.average([rewards[i] for i in elite_idxs])}\t'
                #       f'Average Score: {np.average(rewards)}'
                #       )
                # print(f'{best_hyperparams} with score: {best_score}')
                print(f'{best_score}')

        return func(best_hyperparams), best_hyperparams
