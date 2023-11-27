import numpy.typing as npt
from typing import Optional, Callable, Any

import scipy
import numpy as np

from .base import Optimizer, OptimizationResult


class ScipyOptimizer(Optimizer):
    """
    Class for the SciPy minimizer.

    Attributes
    ----------
    maxfun : int
        Maximum number of function evaluations.
    bounds : list[tuple[float, float]] or None
        A list of tuples specifying the lower and upper bounds for each
        dimension of the search space, or None if no bounds are provided.
    optimizer_kwargs : dict[str, Any]
        Additional keyword arguments for the SciPy minimizer.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    verbose : bool
        If set to True, additional information will be printed (default False).
    """
    def __init__(
            self,
            maxfun: int,
            bounds: Optional[list[tuple[float, float]]] = None,
            optimizer_kwargs: dict[str, Any] = {},
            verbose: bool = False,
    ) -> None:
        self.maxfun = maxfun
        self.bounds = bounds
        self.optimizer_kwargs = optimizer_kwargs
        self.verbose = verbose

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
        init: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        """
        Minimize the given function using the SciPy minimize.

        Parameters
        ----------
        func : callable
            The objective function to be minimized.
            The function should take a single argument, which is a NumPy array
            of type np.float64, and return a float value.
        init : numpy.ndarray
            The initial point for the optimization algorithm.
            The array should have dtype np.float64.

        Returns
        -------
        tuple
            A tuple containing the minimum function value and the
            corresponding optimal point.
        """
        def wrapper(params: npt.NDArray[np.float64]) -> float:
            return func(np.array(params).reshape(np.array(init).shape)).value

        history: list[OptimizationResult] = []

        def callback(intermediate_result):
            if self.verbose:
                print(f"Step {len(history)+1}/{self.maxfun}: "
                      f"{float(intermediate_result.fun)}")
            history.append(OptimizationResult(
                intermediate_result.fun, np.copy(intermediate_result.x)))

        if 'options' not in self.optimizer_kwargs:
            self.optimizer_kwargs['options'] = {}
        if 'maxfun' not in self.optimizer_kwargs['options']:
            self.optimizer_kwargs['options']['maxfun'] = self.maxfun

        result = scipy.optimize.minimize(
            wrapper,
            np.array(init).flatten(),
            bounds=(
                self.bounds if self.bounds is not None
                else [(0, 2*np.pi)]*len(np.array(init).flatten())
            ),
            callback=callback,
            **self.optimizer_kwargs
        )
        
        if self.verbose:
            print(result.success, result.message)
            
        return OptimizationResult(
            result.fun, result.x.reshape(np.array(init).shape),
            history
        )
