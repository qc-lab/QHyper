from typing import Any, Callable

import scipy
import numpy as np

from .optimizer import ArgsType, Optimizer, HyperparametersOptimizer


class ScipyAllOptimizer(HyperparametersOptimizer):
    def __init__(self, maxfun: int, bounds: list[tuple[float, float]]=None) -> None:
        self.maxfun = maxfun
        self.bounds = bounds

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        evaluation_func: Callable[[ArgsType], Callable[[ArgsType], float]] = None,
        bounds: list[float] = None,
        **kwargs: Any
    ) -> ArgsType:
        # def wrapper(params):
        #     return func_creator(hyperparams_init)(np.array(params).reshape(np.array(init).shape))
        # result = scipy.optimize.minimize(
        #     wrapper, np.array(init).flatten(),
        #     bounds=self.bounds[2:],
        #     options = {'maxfun': self.maxfun}
        # )
        # return result.fun, np.array(result.x).reshape(init.shape), hyperparams_init
        def wrapper(params):
            weights = params[:len(hyperparams_init)]
            angles = np.array(params[len(hyperparams_init):]).reshape(init.shape)

            return evaluation_func(weights, angles)

        init_params = list(hyperparams_init) + list(np.array(init).flatten())
        result = scipy.optimize.minimize(
            wrapper, np.array(init_params).flatten(),
            bounds=self.bounds,
            options = {'maxfun': self.maxfun}
        )
        return result.fun, np.array(result.x[len(hyperparams_init):]).reshape(init.shape), result.x[:len(hyperparams_init)]
        # return result.x.reshape(np.array(init).shape), []
