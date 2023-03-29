from scipy.optimize import basinhopping
import numpy as np
import numpy.typing as npt
from typing import Callable, Any

from .base import Optimizer


class Basinhopping(Optimizer):
    """Implementation of Cross Entropy Method for hyperparamter tuning
    """
    def __init__(
            self, 
            bounds: list[tuple[float, float]], 
            niter: int, 
            maxfun: int, 
            config: dict[str, Any]= {}
    ) -> None:
        self.niter = niter
        self.maxfun = maxfun
        self.bounds = np.array(bounds)
        self.config = config

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], float],
        init: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
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
                'bounds': self.bounds
            }, **self.config)

        return result.fun, np.array(result.x).reshape(init.shape)
