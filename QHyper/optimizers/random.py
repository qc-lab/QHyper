import multiprocessing as mp
from typing import Callable

import numpy as np
import tqdm

from .optimizer import ArgsType, HyperparametersOptimizer, Optimizer, Wrapper


class Random(HyperparametersOptimizer):
    """Simple random search
    
    Attributes
    ----------
    number_of_samples : int
        number of random samples (default 100)
    processes : int
         number of processors that will be used (default cpu count)
    disable_tqdm: bool
        if set to True, tdqm will be disabled (default False)
    """

    def __init__(
        self,
        number_of_samples: int = 100,
        processes: int = 1,
        disable_tqdm: bool = False,
        bounds: list[tuple[float, float]] = None
    ) -> None:
        """
        Parameters
        ----------
        number_of_samples : int
            number of random samples (default 100)
        processes : int
            number of processors that will be used (default cpu count)
        disable_tqdm: bool
            if set to True, tdqm will be disabled (default False)
        """

        self.number_of_samples: int = number_of_samples
        self.processes: int = processes
        self.disable_tqdm: bool = disable_tqdm
        self.bounds = np.array(bounds)
    
    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType = None, 
        evaluation_func: Callable[[ArgsType], Callable[[ArgsType], float]] = None,
        # bounds: list[float] = [0, 10]
    ) -> ArgsType:
        """Returns hyperparameters which lead to the lowest values returned by the optimizer
    
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
            Returns hyperparameters which lead to the lowest values returned by the optimizer
        """
        hyperparams_init = np.array(hyperparams_init)

        hyperparams = (
            (self.bounds[:, 1] - self.bounds[:, 0])
            * np.random.rand(self.number_of_samples, *hyperparams_init.shape)
            + self.bounds[:, 0])

        wrapper = Wrapper(func_creator, optimizer, evaluation_func, init)

        with mp.Pool(processes=self.processes) as p:
            results = list(tqdm.tqdm(
                p.imap(wrapper.func, hyperparams), total=self.number_of_samples, disable=self.disable_tqdm))

        min_idx = np.argmin([result[0] for result in results])

        return *wrapper.func(hyperparams[min_idx]), hyperparams[min_idx]
