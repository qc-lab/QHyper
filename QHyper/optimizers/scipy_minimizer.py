from typing import Any, Callable

import scipy
import numpy as np

from .optimizer import ArgsType, Optimizer


class ScipyOptimizer(Optimizer):
    def __init__(self, maxfun: int, bounds: list[tuple[float, float]]=None) -> None:
        self.maxfun = maxfun
        self.bounds = bounds

    def minimize(self, func: Callable[[ArgsType], float], init: ArgsType) -> tuple[ArgsType, Any]:
        def wrapper(params):
            return func(np.array(params).reshape(np.array(init).shape))
        result = scipy.optimize.minimize(
            wrapper, np.array(init).flatten(),
            bounds=self.bounds if self.bounds is not None else [(0, 2*np.pi)]*len(np.array(init).flatten()),
            options = {'maxfun': self.maxfun}
        )
        return result.x.reshape(np.array(init).shape), []
