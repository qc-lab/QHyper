import itertools
import multiprocessing as mp
from dataclasses import dataclass

from typing import Callable
import numpy as np
import numpy.typing as npt
from .base import OptimizationResult, Optimizer
import tqdm


@dataclass
class GridSearch(Optimizer):
    bounds: npt.NDArray[np.float64]
    steps: list[np.float64]
    processes: int = 1
    disable_tqdm: bool = False

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        steps: list[float],
        processes: int = 1,
        disable_tqdm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        bounds : list[tuple[float, float]]
            list of tuples with lower and upper bounds for each variable
        steps : list[float]
            step for each variable bound
        processes : int
            number of processors that will be used (default cpu count)
        disable_tqdm: bool
            if set to True, tdqm will be disabled (default False)
        """

        self.bounds = np.array(bounds)
        self.steps = steps
        self.processes = processes
        self.disable_tqdm = disable_tqdm

    def generate_grid(self):
        hyperparams = []
        for params in itertools.product(*[
                np.arange(*bound, step)
                for bound, step in zip(self.bounds, self.steps)
        ]):
            hyperparams.append(np.array(params))
        return hyperparams

    def minimize(
            self,
            func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
            init: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        hyperparams = self.generate_grid()

        with mp.Pool(processes=self.processes) as p:
            results = list(tqdm.tqdm(
                p.imap(func, hyperparams),
                total=len(hyperparams),
                disable=self.disable_tqdm
            ))
        min_idx = np.argmin([result.value for result in results])

        return OptimizationResult(
            value=results[min_idx].value,
            params=hyperparams[min_idx],
        )
