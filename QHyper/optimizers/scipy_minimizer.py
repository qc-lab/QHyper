from typing import Any, Callable

import scipy
import numpy as np

from .optimizer import ArgsType, Optimizer


class ScipyOptimizer(Optimizer):
    def minimize(self, func: Callable[[ArgsType], float], init: ArgsType) -> tuple[ArgsType, Any]:
        def wrapper(params):
            return func(np.array(params).reshape(np.array(init).shape))
        result = scipy.optimize.minimize(wrapper, np.array(init).flatten())
        return result.x.reshape(np.array(init).shape), []
