import multiprocessing as mp
import numpy as np
import tqdm

from dataclasses import dataclass
from typing import Callable, Any

from .optimizer import HyperparametersOptimizer, Worker, ArgsType, Optimizer


@dataclass
class Random(HyperparametersOptimizer):
    """Simple random search
    
    Args:
        - number_of_samples - an integer indicating amount of random samples as hyperparameters (default 100)
        - processes - an integer indicating how many processors will be used (default cpu count)
    """
    number_of_samples: int = 100
    processes: int = mp.cpu_count()

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType = None, 
        bounds: list[float] = [0, 10],
        **kwargs: Any
    ) -> ArgsType:
        """This method receives:
            - func_creator - function, which receives hyperparameters, and returns 
                function which will be optimized using optimizer
            - optimizer - object of class Optimizer
            - init - initial args for optimizer
            - hyperparams_init - initial hyperparameters, only needed to get shape of args
            - bounds - bounds for hyperparameters
            - kwargs - allow additional arguments, although these method doesn't use any


        Returns hyperparameters which leads to the lowest values returned by optimizer
        """
        hyperparams_init = np.array(hyperparams_init)

        hyperparams = (
            (bounds[1] - bounds[0]) 
            * np.random.rand(self.number_of_samples, *hyperparams_init.shape)
            + bounds[0])

        worker = Worker(func_creator, optimizer, init)

        with mp.Pool(processes=self.processes) as p:
            results = list(tqdm.tqdm(p.imap(worker.func, hyperparams), total=self.number_of_samples))
            
        min_idx = np.argmin([result for result in results])

        return hyperparams[min_idx]
