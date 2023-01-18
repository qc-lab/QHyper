from typing import Any

from .optimizer import ArgsType, HyperparametersOptimizer


class Static(HyperparametersOptimizer):
    """Simple random search
    
    Attributes
    ----------
    weights: list[float]
        weights, which will be used to create QUBO
    """

    def __init__(
        self,
        weights: list[float]
    ) -> None:
        """
        Parameters
        ----------
        weights: list[float]
            weights, which will be used to create QUBO
        """

        self.weights: list[float] = weights
    
    def minimize(
        self, 
        **kwargs: Any
    ) -> ArgsType:
        """Returns weights
        """
        return self.weights
