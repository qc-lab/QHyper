from dataclasses import dataclass
import multiprocessing as mp
from typing import Callable
import numpy.typing as npt

import numpy as np
import tqdm

from .base import Optimizer, OptimizationResult


@dataclass
class Random(Optimizer):
    number_of_samples: int
    processes: int
    disable_tqdm: bool
    bounds: npt.NDArray[np.float64]
    verbose: bool = False

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        number_of_samples: int = 100,
        processes: int = 1,
        disable_tqdm: bool = False,
        verbose: bool = False,
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
        verbose: bool
            if set to True, additional information will be printed
            (default False)
        """

        self.number_of_samples: int = number_of_samples
        self.processes: int = processes
        self.disable_tqdm: bool = disable_tqdm
        self.bounds = np.array(bounds)
        self.verbose = verbose

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
        init: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        """Returns hyperparameters which lead to the lowest values
            returned by the optimizer

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
            Returns hyperparameters which lead to the lowest values
            returned by the optimizer
        """
        _init = np.array(init)
        hyperparams = (
            (self.bounds[:, 1] - self.bounds[:, 0])
            * np.random.rand(self.number_of_samples, *_init.flatten().shape)
            + self.bounds[:, 0])

        with mp.Pool(processes=self.processes) as p:
            results = list(tqdm.tqdm(
                p.imap(func, [h.reshape(_init.shape) for h in hyperparams]),
                total=self.number_of_samples,
                disable=self.disable_tqdm
            ))
        min_idx = np.argmin([result.value for result in results])

        if self.verbose:
            print(f"Best result: {results[min_idx].value}")
            print(f"Values: {sorted([v.value for v in results])}")

        history = [OptimizationResult(res.value, params, [[res]])
                   for res, params in zip(results, hyperparams)]

        return OptimizationResult(
            value=results[min_idx].value,
            params=hyperparams[min_idx],
            history=[history],
        )
