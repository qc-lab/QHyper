from abc import ABC, abstractmethod
from typing import Any, Callable

ArgsType = float | list[float]


class Optimizer(ABC):
    """Interface for optimizers"""

    @abstractmethod
    def minimize(
        self,
        func: Callable[[ArgsType], float],
        init: ArgsType
    ) -> tuple[ArgsType, Any]:
        """Returns params which lead to the lowest value of the provided function and cost history

        Parameters
        ----------
        func : Callable[[ArgsType], float]
            function, which will be minimized
        init : ArgsType
            initial args for the optimizer

        Returns
        -------
        tuple[ArgsType, Any]
            Returns tuple which contains params that lead to the lowest value
            of the provided function and cost history
        """
        pass


class HyperparametersOptimizer(ABC):
    """Interface for hyperoptimizers"""

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer, 
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        evaluation_func: Callable[[ArgsType], Callable[[ArgsType], float]],
        bounds: list[float],
        **kwargs: Any
    ) -> ArgsType:
        """Returns hyperparameters which lead to the lowest values returned by the optimizer
        
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
            bounds for hyperparameters
        kwargs : Any
            allow additional arguments

        Returns
        -------
        ArgsType
            hyperparameters which lead to the lowest values returned by the optimizer
        """
        pass


class Wrapper:
    """Use hyperparameters to create a function using func_creator,
    run optimizer on this function, and return its lowest value

    Usefull for HyperparametersOptimizer, which uses multiprocessing, as local functions are not
    easy pickleable.
    """

    def __init__(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]],  
        optimizer: Optimizer, 
        evaluation_func: Callable[[ArgsType, ArgsType], float],
        init: ArgsType
    ) -> None:
        self.func_creator = func_creator
        self.optimizer = optimizer
        self.evaluation_func = evaluation_func
        self.init = init

    def func(self, hyperparams: Any) -> float:
        _func = self.func_creator(hyperparams)
        params, _ = self.optimizer.minimize(_func, self.init)
        return self.evaluation_func(hyperparams, params)
