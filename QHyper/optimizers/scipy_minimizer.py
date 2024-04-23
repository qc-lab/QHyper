# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Callable, Any

import scipy
import numpy as np

from .base import Optimizer, OptimizationResult, OptimizerError


@dataclass
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
    verbose : bool
        If set to True, additional information will be printed (default False).
    kwargs : dict[str, Any]
        Additional keyword arguments for the SciPy minimizer.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    maxfun: int
    kwargs: dict[str, Any] = field(default_factory=dict)

    def _minimize(
        self,
        func: Callable[[NDArray], OptimizationResult],
        init: NDArray
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
        if self.bounds is None:
            raise OptimizerError("This optimizer requires bounds")

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
