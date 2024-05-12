# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from numpy.typing import NDArray
from typing import Callable, Any

import scipy
import numpy as np

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError)


class ScipyOptimizer(Optimizer):
    """
    Class for the SciPy minimizer.

    Attributes
    ----------
    bounds : numpy.ndarray, optional
        The bounds for the optimization algorithm. Not all optimizers
        support bounds. The shape of the array should be (n, 2), where
        n is the number of parameters (`init` in method :meth:`minimize`).
    verbose : bool, default False
        Whether to print the optimization progress.
    disable_tqdm : bool, default True
        Whether to disable the tqdm progress bar.
    maxfun : int
        Maximum number of function evaluations.
    kwargs : dict[str, Any]
        Additional keyword arguments for the SciPy minimizer.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    bounds: NDArray
    verbose: bool
    disable_tqdm: bool
    maxfun: int
    kwargs: dict[str, Any]

    def __init__(
        self,
        bounds: NDArray,
        verbose: bool = False,
        disable_tqdm: bool = True,
        maxfun: int = 200,
        **kwargs: Any
    ) -> None:
        self.bounds = bounds
        self.verbose = verbose
        self.disable_tqdm = disable_tqdm
        self.maxfun = maxfun
        self.kwargs = kwargs

    def minimize_(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray | None
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
        if init is None:
            raise OptimizerError("Initial point must be provided.")
        self.check_bounds(init)

        def wrapper(params: NDArray) -> float:
            return func(params).value

        history: list[OptimizationResult] = []

        def callback(intermediate_result):
            if self.verbose:
                print(f"Step {len(history)+1}/{self.maxfun}: "
                      f"{float(intermediate_result.fun)}")
            history.append(OptimizationResult(
                intermediate_result.fun, np.copy(intermediate_result.x)))

        if 'options' not in self.kwargs:
            self.kwargs['options'] = {}
        if 'maxfun' not in self.kwargs['options']:
            self.kwargs['options']['maxfun'] = self.maxfun

        result = scipy.optimize.minimize(
            wrapper, init,
            bounds=(self.bounds),
            callback=callback,
            **self.kwargs
        )
        if self.verbose:
            print(f"Success: {result.success}. Message: {result.message}")

        return OptimizationResult(
            result.fun, result.x.reshape(np.array(init).shape), [history]
        )
