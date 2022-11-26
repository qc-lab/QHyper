import numpy as np

import multiprocessing as mp
import tqdm 
from typing import Callable, Any

from .optimizer import HyperparametersOptimizer, Worker, ArgsType, Optimizer


class CEM(HyperparametersOptimizer):
    """Implementation of Cross Entropy Method for hyperparamter tuning
    
    Attributes
    ----------
    epochs : int
        number of epochs (default 10)
    samples_per_epoch : int
        number of samples in each epoch (default 100)
    elite_frac : float
        indicate how many top samples will be used to calculate 
        mean and cov for next epoch - percent (default 0.1)
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
            indicate how many top samples will be used to calculate 
            mean and cov for next epoch (default 0.1)
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
        bounds: list[float] = None,
        **kwargs: Any
    ) -> ArgsType:
        """Returns hyperparameters which leads to the lowest values returned by optimizer 1
        
        Parameters
        ----------
        func_creator : Callable[[ArgsType], Callable[[ArgsType], float]]
            function, which receives hyperparameters, and returns  
            function which will be optimized using optimizer
        optimizer : Optimizer
            object of class Optimizer
        init : ArgsType
            initial args for optimizer
        hyperparams_init : ArgsType
            initial hyperparameters
        bounds : list[float]
            bounds for hyperparameters (default None)
        kwargs : Any
            allow additional arguments, available args:
                print_freq : int
                    prints additional score information every x epochs

        Returns
        -------
        ArgsType
            hyperparameters which leads to the lowest values returned by optimizer
        """
        mean = [1] * len(hyperparams_init) 
        cov = np.identity(len(hyperparams_init))
        best_hyperparams = hyperparams_init
        best_score = float('inf')

        print_freq = kwargs.get("print_freq", self.epochs+1)

        scores = []
        worker = Worker(func_creator, optimizer, init)
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
            cov = np.cov(np.stack((elite_weights), axis = 1), bias=True)

            if i_iteration % print_freq == 0:
                print(f'Epoch {i_iteration}\t'
                      f'Average Elite Score: {np.average([rewards[i] for i in elite_idxs])}\t'
                      f'Average Score: {np.average(rewards)}'
                )
                print(f'{best_hyperparams} with score: {best_score}')
        return best_hyperparams
