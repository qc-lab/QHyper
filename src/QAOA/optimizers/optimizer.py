from abc import ABC, abstractmethod
from typing import Callable, Any


ArgsType = float | list[float]


class Optimizer(ABC):
    """Abstract class for optimizers
    
    Methods
    -------
    minimize(func, init):
        Returns params which leads to the lowest value of the provided function
    """

    @abstractmethod
    def minimize(
        self, 
        func: Callable[[ArgsType], float], 
        init: ArgsType
    ) -> ArgsType:
        """Returns params which leads to the lowest value of the provided function 

        Parameters
        ----------
        func : Callable[[ArgsType], float]
            function, which will be minimize
        init : ArgsType
            initial args for optimizer

        Returns
        -------
        ArgsType
            Returns args which gave the lowest value.
        """
        pass


class HyperparametersOptimizer(ABC):
    """Abstract class for hyperoptimizers
    
    Methods
    -------
    minimize(func_creator, optimizer, init, hyperparams_init, bounds, **kwargs):
        Returns hyperparameters which leads to the lowest values returned by optimizer
    """


    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer, 
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        bounds: list[float],
        **kwargs: Any
    ) -> ArgsType:
        """Returns hyperparameters which leads to the lowest values returned by optimizer 
        
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
            bounds for hyperparameters
        kwargs : Any
            allow additional arguments

        Returns
        -------
        ArgsType
            hyperparameters which leads to the lowest values returned by optimizer
        """
        pass


class Worker: #change name
    """This class has one purpose, implement method, which recevies hyperparameters,
    create function using func_creator, run optimizer on this function, and returns its lowest value

    Usefull with HyperparametersOptimizer, which uses multiprocessing, as local functions are not 
    easy picklable.
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
