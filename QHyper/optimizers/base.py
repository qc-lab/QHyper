# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import abc
import warnings
import dataclasses

from abc import abstractmethod
import numpy as np

from typing import Callable, cast, overload
from numpy.typing import NDArray


class OptimizerError(Exception):
    """
    Base class for exceptions in this module.
    """
    ...


@dataclasses.dataclass
class OptimizationResult:
    """
    Dataclass for storing the results of an optimization run.

    Attributes
    ----------
    value : float
        The minimum function value found by the optimization algorithm.
    params : numpy.ndarray
        The optimal point found by the optimization algorithm.
    history : list[list[OptimizationResult]]
        The history of the optimization algorithm. Each element of the list
        represents the results of the objective function at each
        iteration - there can be multiple results per each iteration (epoch).
    """

    value: float
    params: NDArray
    history: list[list['OptimizationResult']] = dataclasses.field(
        default_factory=list)

    def fix_dims(self, dims: tuple[int]) -> 'OptimizationResult':
        self.params = self.params.reshape(dims)

        for epoch in self.history:
            for result in epoch:
                result.params = result.params.reshape(dims)
        return self


class Optimizer(abc.ABC):
    """
    Base class for Optimizer.

    """

    @overload
    def minimize(self, func: Callable[[NDArray], OptimizationResult]
                 ) -> OptimizationResult:
        ...

    @overload
    def minimize(self, func: Callable[[NDArray], OptimizationResult],
                 init: NDArray) -> OptimizationResult:
        ...

    def minimize(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray | None = None
    ) -> OptimizationResult:
        """
        Method that minimizes the given function using the
        implemented optimization algorithm. This method checks
        the arguments and calls the abstract method :py:meth:`_minimize`.

        Parameters
        ----------
        func : callable
            The objective function to be minimized.
        init : numpy.ndarray
            The initial point for the optimization algorithm.
            Some algorithms requires init just to obtain the shape.

        Returns
        -------
        tuple
            A tuple containing the minimum function value and the
            corresponding optimal point.
        """
        if init is None:
            return self.minimize_(func, None)

        _init = np.copy(init).flatten()

        result = self.minimize_(func, _init)
        return result.fix_dims(cast(tuple[int], init.shape))

    def check_bounds(self, init: NDArray | None) -> None:
        """
        Check if the bounds are correctly set. This method should be
        called before the optimization starts.
        """
        if not hasattr(self, "bounds"):
            raise OptimizerError("This optimizer requires bounds.")

        bounds = getattr(self, "bounds")
        if isinstance(bounds, list):
            # warnings.warn("WARNING: bounds should be a numpy array. "
            #               "Converting to numpy array.")
            setattr(self, "bounds", np.array(bounds))
            bounds = getattr(self, "bounds")

        if bounds.shape[-1] != 2:
            raise OptimizerError("Bounds should be a 2D "
                                 "array with shape (n, 2).")

        if init is not None and bounds.shape[:-1] != init.shape:
            raise OptimizerError(
                f"Bounds shape {bounds.shape[:-1]} "
                f"does not match init shape {init.shape}."
            )

    @abstractmethod
    def minimize_(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray | None
    ) -> OptimizationResult:
        """
        Abstract method that should be implemented by the subclass.
        """
