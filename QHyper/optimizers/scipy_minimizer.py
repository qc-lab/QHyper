import numpy.typing as npt
from typing import Optional, Callable


import scipy
import numpy as np

from .base import Optimizer


class ScipyOptimizer(Optimizer):
    def __init__(
            self, 
            maxfun: int, 
            bounds: Optional[list[tuple[float, float]]] = None
    ) -> None:
        self.maxfun = maxfun
        self.bounds = bounds

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], float],
        init: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        def wrapper(params: npt.NDArray[np.float64]) -> float:
            return func(np.array(params).reshape(np.array(init).shape))

        result = scipy.optimize.minimize(
            wrapper, np.array(init).flatten(),
            bounds=self.bounds if self.bounds is not None else [(0, 2*np.pi)]*len(np.array(init).flatten()),
            options = {'maxfun': self.maxfun}
        )
        return result.fun, result.x.reshape(np.array(init).shape)
