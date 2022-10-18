import multiprocessing as mp
import numpy as np
import tqdm

from dataclasses import dataclass
from typing import Callable

from .optimizer import HyperparametersOptimizer, Worker, ArgsType, Optimizer


@dataclass
class Random(HyperparametersOptimizer):
    number_of_samples: int = 100
    process: int = mp.cpu_count()

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType = None, 
        bounds: list[float] = [0, 10],
    ) -> ArgsType:
        hyperparams_init = np.array(hyperparams_init)

        hyperparams = (
            (bounds[1] - bounds[0]) 
            * np.random.rand(self.number_of_samples, *hyperparams_init.shape)
            + bounds[0])

        worker = Worker(func_creator, optimizer, init)

        with mp.Pool(processes=self.process) as p:
            results = list(tqdm.tqdm(p.imap(worker.func, hyperparams), total=self.number_of_samples))
            
        min_idx = np.argmin([result for result in results])

        return hyperparams[min_idx]
