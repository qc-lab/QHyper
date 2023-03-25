from scipy.optimize import basinhopping
import numpy as np

from typing import Callable, Any

from .base import Optimizer


class Basinhopping(Optimizer):
    """Implementation of Cross Entropy Method for hyperparamter tuning
    """
    def __init__(self, niter: int, maxfun: int, bounds: list[tuple[float, float]] = None, config: dict[str, Any]= {}) -> None:
        self.niter = niter
        self.maxfun = maxfun
        self.bounds = np.array(bounds)
        self.config = config

    def minimize(
        self,
        func: Callable[[list[float]], float],
        init: list[float]
    ) -> tuple[float, list[float]]:
        """Returns hyperparameters which leads to the lowest values returned by optimizer 1
        
        Parameters
        ----------
        func_creator : Callable[[ArgsType], Callable[[ArgsType], float]]
            function, which receives hyperparameters, and returns  
            function which will be optimized using optimizer
        optimizer : Optimizer
            object of class Optimizer
        init : ArgsType
            initial args for optimizer
        hyperparams_init : ArgsType
            initial hyperparameters
        bounds : list[float]
            bounds for hyperparameters (default None)
        evaluation_func : Callable[[ArgsType], Callable[[ArgsType], float]]
            function, which receives hyperparameters, and returns 
            function which receives params and return evaluation
        kwargs : Any
            allow additional arguments passed to scipy.optimize.Shgo
        Returns
        -------
        ArgsType
            hyperparameters which leads to the lowest values returned by optimizer
        """

        result = basinhopping(
            func, init, niter=self.niter, 
            minimizer_kwargs={
                'options': {'maxfun': self.maxfun},
                'bounds': self.bounds[2:]
            }, **self.config)

        return result.fun, np.array(result.x).reshape(init.shape)
