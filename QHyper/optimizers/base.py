# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import dataclasses

from abc import abstractmethod
import numpy as np

from typing import Callable, Optional, cast
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
    history : list[list[float]]
        The history of the optimization algorithm. Each element of the list
        represents the values of the objective function at each
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


@dataclasses.dataclass(kw_only=True)
class Optimizer:
    bounds: Optional[NDArray] = None
    verbose: bool = False
    disable_tqdm: bool = True

    def __post_init__(self):
        if self.bounds is not None and not isinstance(self.bounds, np.ndarray):
            self.bounds = np.array(self.bounds)

    def minimize(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray
    ) -> OptimizationResult:
        """
        Abstract method that minimizes the given function using the
        implemented optimization algorithm.

        Parameters
        ----------
        func : callable
            The objective function to be minimized.
        init : numpy.ndarray
            The initial point for the optimization algorithm.

        Returns
        -------
        tuple
            A tuple containing the minimum function value and the
            corresponding optimal point.
        """
        _init = np.copy(init).flatten()

        if _init.ndim != 1:
            raise OptimizerError("Init should be a 1D array.")
        if self.bounds is not None:
            if self.bounds.shape[-1] != 2:
                raise OptimizerError("Bounds should be a 2D "
                                     "array with shape (n, 2).")

            if self.bounds.shape[:-1] != _init.shape:
                raise OptimizerError(
                    f"Bounds shape {self.bounds.shape[:-1]} "
                    f"does not match init shape {_init.shape}."
                )
        result = self._minimize(func, _init)
        return result.fix_dims(cast(tuple[int], init.shape))

    @abstractmethod
    def _minimize(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray
    ) -> OptimizationResult:
        ...
