import multiprocessing as mp
from typing import Any, Callable

import numpy as np
import tqdm

from .optimizer import ArgsType, HyperparametersOptimizer, Optimizer, Wrapper


class AllCEM(HyperparametersOptimizer):
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

    def __init__(
        self, epochs: int = 10,
        samples_per_epoch: int = 100,
        elite_frac: float = 0.1,
        processes: int = mp.cpu_count(),
        print_on_epochs: list[int] = [],
        disable_tqdm: bool = False,
        bounds: list[tuple[float, float]] = None
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
        print_on_epochs: list[int]
            list indicating after which epochs print best results (default [])
        disable_tqdm: bool
            if set to True, tdqm will be disabled (default False)
        """

        self.epochs: int = epochs
        self.samples_per_epoch: int = samples_per_epoch
        self.elite_frac: float = elite_frac
        self.processes: int = processes
        self.n_elite: int = max(int(self.samples_per_epoch * self.elite_frac), 1)
        self.print_on_epochs: list[int] = print_on_epochs
        self.disable_tqdm = disable_tqdm
        self.bounds = np.array(bounds)

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        evaluation_func: Callable[[ArgsType], Callable[[ArgsType], float]] = None,
        # bounds: list[float] = None
    ) -> ArgsType:
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

        wrapper = Wrapper(func_creator, optimizer, evaluation_func, None)
        # mean = [1] * (len(hyperparams_init)+len(np.array(init).flatten()))
        mean = np.concatenate((hyperparams_init, np.array(init).flatten()))
        cov = np.identity(len(hyperparams_init)+len(np.array(init).flatten()))
        best_hyperparams = hyperparams_init
        best_angles = init
        best_score, _ = wrapper.func(hyperparams_init, init)

        for i_iteration in range(1, self.epochs+1):
            params = []
            while len(params) < self.samples_per_epoch:
                point = np.random.multivariate_normal(mean, cov)
                if (self.bounds[:, 0] <= point).all() and (point < self.bounds[:, 1]).all(): 
                    params.append(point)
            params = np.array(params)
            # params = np.random.multivariate_normal(mean, cov, size=self.samples_per_epoch)
            hyperparams = params[:, :len(hyperparams_init)]
            angles = np.array(params[:, len(hyperparams_init):]).reshape((-1, *np.array(init).shape))
            # print(hyperparams)
            # print(angles)
            # if bounds:
                # hyperparams[hyperparams < bounds[0]] = bounds[0]
                # hyperparams[hyperparams > bounds[1]] = bounds[1]

            # hyperparams = np.concatenate((hyperparams, [best_hyperparams.flatten()]), axis=0)
            # hyperparams = [
                # np.reshape(np.array(hyperparam), hyperparams_init.shape) for hyperparam in hyperparams]

            with mp.Pool(processes=self.processes) as p:
                results = list(tqdm.tqdm(
                    p.starmap(wrapper.func, zip(hyperparams, angles)), total=len(hyperparams), disable=self.disable_tqdm))

            rewards = np.array([result[0] for result in results])

            elite_idxs = rewards.argsort()[:self.n_elite]
            elite_weights = [hyperparams[i].flatten() for i in elite_idxs]
            elite_angles = [angles[i].flatten() for i in elite_idxs]

            # best_hyperparams = elite_weights[0].reshape(hyperparams_init.shape)
            # best_angles = elite_angles[0].reshape(np.array(init).shape)

            # reward, _ = wrapper.func(best_hyperparams)
            if rewards[elite_idxs[0]] < best_score:
                best_hyperparams = elite_weights[0].reshape(hyperparams_init.shape)
                best_angles = elite_angles[0].reshape(np.array(init).shape)
                best_score = rewards[elite_idxs[0]]
            elite_params = np.concatenate([elite_weights, elite_angles], axis=1)
            # scores.append(rewards[elite_idxs[0]])
            mean = np.mean(elite_params, axis=0)
            cov = np.cov(np.stack((elite_params), axis=1), bias=True)

            if i_iteration in self.print_on_epochs:
                # print(f'Epoch {i_iteration}\t'
                #       f'Average Elite Score: {np.average([rewards[i] for i in elite_idxs])}\t'
                #       f'Average Score: {np.average(rewards)}'
                #       )
                # print(f'{best_hyperparams} with score: {best_score}')
                print(f'{best_score}')

        return best_score, best_angles, best_hyperparams
