from abc import ABC, abstractmethod
from typing import Callable, Any


ArgsType = float | list[float]


class Optimizer(ABC):
    @abstractmethod
    def minimize(
        self, 
        func: Callable[[ArgsType], float], 
        init: ArgsType
    ) -> ArgsType:
        """Find minimum of the provided function

        This method receives a function to minimize, and initial args for the function.
        Returns args which gave the lowest value.
        """
        pass


class HyperparametersOptimizer(ABC):
    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer, 
        init: ArgsType, 
        hyperparams_init: ArgsType, 
        bounds: list[float]
    ) -> ArgsType:
        """Find best hyperparameters

        This method receives:
            - func_creator - function, which receives hyperparameters, and returns 
                function which will be optimized using optimizer
            - optimizer - object of class Optimizer
            - init - initial args for optimizer
            - hyperparams_init - initial hyperparameters
            - bounds - bounds for hyperparameters

        Returns hyperparameters which leads to the lowest values returned by optimizer
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
