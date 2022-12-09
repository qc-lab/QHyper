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
    ) -> ArgsType:
        """Returns params which lead to the lowest value of the provided function

        Parameters
        ----------
        func : Callable[[ArgsType], float]
            function, which will be minimized
        init : ArgsType
            initial args for the optimizer

        Returns
        -------
        ArgsType
            Returns args which gave the lowest value
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


class Worker:  # change name
    """Use hyperparameters to create a function using func_creator,
    run optimizer on this function, and return its lowest value

    Usefull for HyperparametersOptimizer, which uses multiprocessing, as local functions are not
    easy pickleable.
    """

    def __init__(
            self,
            func_creator: Callable[[ArgsType], Callable[[ArgsType], float]],
            optimizer: Optimizer,
            init: ArgsType
    ) -> None:
        self.func_creator = func_creator
        self.optimizer = optimizer
        self.init = init

    def func(self, hyperparams: Any) -> float:
        _func = self.func_creator(hyperparams)
        params = self.optimizer.minimize(_func, self.init)
        return _func(params)
