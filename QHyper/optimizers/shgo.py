from scipy.optimize import shgo
import numpy as np

from typing import Callable, Any

from .optimizer import HyperparametersOptimizer, ArgsType, Optimizer, Wrapper


class Shgo(HyperparametersOptimizer):
    """Implementation of Cross Entropy Method for hyperparamter tuning
    """
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

        # wrapper = Wrapper(func_creator, optimizer, evaluation_func, init)
        def wrapper(params):
            weights = params[:len(hyperparams_init)]
            angles = np.array(params[len(hyperparams_init):]).reshape(init.shape)

            return evaluation_func(weights, angles)

        init_params = list(hyperparams_init) + list(np.array(init).flatten())

        result = shgo(
            wrapper,
            bounds=[[0.01, 10], [0.01, 10], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None], [None, None],],
            **kwargs)

        return result.fun, np.array(result.x[len(hyperparams_init):]).reshape(init.shape), result.x[:len(hyperparams_init)]
