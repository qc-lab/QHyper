import multiprocessing as mp
from typing import Any, Callable

import numpy as np
import tqdm

from .optimizer import ArgsType, HyperparametersOptimizer, Optimizer, Worker


class CEM(HyperparametersOptimizer):
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
    """

    def __init__(
            self, epochs: int = 10,
            samples_per_epoch: int = 100,
            elite_frac: float = 0.1,
            processes: int = mp.cpu_count()
    ) -> None:
        """
        Parameters
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
        """

        self.epochs: int = epochs
        self.samples_per_epoch: int = samples_per_epoch
        self.elite_frac: float = elite_frac
        self.processes: int = processes
        self.n_elite: int = int(self.samples_per_epoch * self.elite_frac)

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        evaluation_func: Callable[[ArgsType], Callable[[ArgsType], float]] = None,
        bounds: list[float] = None,
        **kwargs: Any
    ) -> ArgsType:
        """Returns hyperparameters which lead to the lowest values returned by optimizer 1
        
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
        kwargs : Any
            allow additional arguments, available args:
                print_freq : int
                    prints additional score information every x epochs

        Returns
        -------
        ArgsType
            hyperparameters which lead to the lowest values returned by the optimizer
        """
        mean = [1] * len(hyperparams_init)
        cov = np.identity(len(hyperparams_init))
        best_hyperparams = hyperparams_init
        best_score = float('inf')

        print_freq = kwargs.get("print_freq", self.epochs + 1)

        scores = []
        worker = Worker(func_creator, optimizer, evaluation_func, init)
        for i_iteration in range(1, self.epochs+1):
            hyperparams = np.random.multivariate_normal(mean, cov, size=self.samples_per_epoch)
            if bounds:
                hyperparams[hyperparams < bounds[0]] = bounds[0]
                hyperparams[hyperparams > bounds[1]] = bounds[1]

            hyperparams = np.concatenate((hyperparams, [best_hyperparams.flatten()]), axis=0)
            hyperparams = [
                np.reshape(np.array(hyperparam), hyperparams_init.shape) for hyperparam in hyperparams]
            rewards = []

            with mp.Pool(processes=self.processes) as p:
                results = list(tqdm.tqdm(p.imap(worker.func, hyperparams), total=len(hyperparams)))

            rewards = np.array([result for result in results])

            elite_idxs = rewards.argsort()[:self.n_elite]
            elite_weights = [hyperparams[i].flatten() for i in elite_idxs]

            best_hyperparams = elite_weights[0].reshape(hyperparams_init.shape)

            reward = worker.func(best_hyperparams)
            if reward < best_score:
                best_hyperparams = best_hyperparams
                best_score = reward

            scores.append(reward)
            mean = np.mean(elite_weights, axis=0)
            cov = np.cov(np.stack((elite_weights), axis=1), bias=True)

            if i_iteration % print_freq == 0:
                print(f'Epoch {i_iteration}\t'
                      f'Average Elite Score: {np.average([rewards[i] for i in elite_idxs])}\t'
                      f'Average Score: {np.average(rewards)}'
                      )
                print(f'{best_hyperparams} with score: {best_score}')
        return best_hyperparams
