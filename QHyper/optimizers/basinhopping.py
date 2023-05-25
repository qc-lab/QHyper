from scipy.optimize import basinhopping
import numpy as np
import numpy.typing as npt
from typing import Callable, Any

from .base import Optimizer


class Basinhopping(Optimizer):
    """
    Class for the Basin-hopping algorithm for global optimization.

    Parameters
    ----------
    bounds : list[tuple[float, float]]
        A list of tuples specifying the lower and upper bounds for each dimension of the search space.
    niter : int
        The number of basin-hopping iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations.
        Default is 200.
    config : dict, optional
        Additional configuration options for the basinhopping function.
        Default is an empty dictionary.

    Attributes
    ----------
    niter : int
        The number of basin-hopping iterations to perform.
    maxfun : int
        Maximum number of function evaluations.
    bounds : numpy.ndarray
        An array of shape (n, 2) specifying the lower and upper bounds for each dimension of the search space.
    config : dict
        Additional configuration options for the basinhopping function.

    Methods
    -------
    minimize(func, init)
        Minimizes the given function using the Basin-hopping algorithm.
    """
    def __init__(
            self,
            bounds: list[tuple[float, float]],
            niter: int,
            maxfun: int = 200,
            config: dict[str, Any] = {}
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
        """
        Minimize the given function using the Basin-hopping algorithm.

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
        result = basinhopping(
            func, init.flatten(), niter=self.niter,
            minimizer_kwargs={
                'options': {'maxfun': self.maxfun},
                'bounds': self.bounds
            }, **self.config)

        return result.fun, np.array(result.x).reshape(init.shape)
