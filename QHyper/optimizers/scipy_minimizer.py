import numpy.typing as npt
from typing import Optional, Callable

import scipy
import numpy as np

from .base import Optimizer


class ScipyOptimizer(Optimizer):
    """
    Class for the SciPy minimizer.

    Attributes
    ----------
    maxfun : int
        Maximum number of function evaluations.
    bounds : list[tuple[float, float]] or None
        A list of tuples specifying the lower and upper bounds for each dimension of the search space,
        or None if no bounds are provided.
    """
    def __init__(
            self,
            maxfun: int,
            bounds: Optional[list[tuple[float, float]]] = None
    ) -> None:
        self.maxfun = maxfun
        self.bounds = bounds

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], float],
        init: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """
        Minimize the given function using the SciPy minimize.

        Parameters
        ----------
        func : callable
            The objective function to be minimized.
            The function should take a single argument, which is a NumPy array of type np.float64,
            and return a float value.
        init : numpy.ndarray
            The initial point for the optimization algorithm.
            The array should have dtype np.float64.

        Returns
        -------
        tuple
            A tuple containing the minimum function value and the corresponding optimal point.
        """
        def wrapper(params: npt.NDArray[np.float64]) -> float:
            return func(np.array(params).reshape(np.array(init).shape))

        result = scipy.optimize.minimize(
            wrapper,
            np.array(init).flatten(),
            bounds=(
                self.bounds if self.bounds is not None
                else [(0, 2*np.pi)]*len(np.array(init).flatten())
            ),
            options={'maxfun': self.maxfun}
        )
        return result.fun, result.x.reshape(np.array(init).shape)
