# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import abc
import warnings
import dataclasses

from abc import abstractmethod
import numpy as np

from typing import Callable, cast, overload, Self
from numpy.typing import NDArray


@dataclasses.dataclass
class OptimizationParameter:
    min: list[float] = dataclasses.field(default_factory=list)
    max: list[float] = dataclasses.field(default_factory=list)
    step: list[float] = dataclasses.field(default_factory=list)
    init: list[float] = dataclasses.field(default_factory=list)

    def assert_bounds(self) -> None:
        if not self.min:
            raise ValueError("Min bounds are required")
        if not self.max:
            raise ValueError("Max bounds are required")
        if len(self.min) != len(self.max):
            raise ValueError("Min and Max bounds must have the same length")

    def assert_step(self) -> None:
        self.assert_bounds()
        if not self.step:
            raise ValueError("Steps are required")
        if len(self.min) != len(self.step):
            raise ValueError("Steps must have the same length as bounds")

    def assert_init(self) -> None:
        if not self.init:
            raise ValueError("Init are required")

    def assert_bounds_init(self) -> None:
        self.assert_bounds()
        self.assert_init()
        if len(self.min) != len(self.init):
            raise ValueError("Init must have the same length as bounds")

    def __add__(self, other: Self) -> Self:
        min_ = self.min + other.min
        max_ = self.max + other.max
        step_ = self.step + other.step
        init_ = self.init + other.init
        return OptimizationParameter(min_, max_, step_, init_)

    def __len__(self) -> int:
        if self.min:
            return len(self.min)
        if self.max:
            return len(self.max)
        if self.init:
            return len(self.init)
        return 0

    def update(self,
               min: list[float] | None = None,
               max: list[float] | None = None,
               step: list[float] | None = None,
               init: list[float] | None = None) -> Self:
        if min is None:
            min = self.min.copy()
        if max is None:
            max = self.max.copy()
        if step is None:
            step = self.step.copy()
        if init is None:
            init = self.init.copy()
        return OptimizationParameter(min, max, step, init)

    @property
    def bounds(self) -> NDArray:
        return list(zip(self.min, self.max))


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
    params: list[float]
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

    # @overload
    # def minimize(self, func: Callable[[list[float]], OptimizationResult]
    #              ) -> OptimizationResult:
    #     ...

    # @overload
    # def minimize(self, func: Callable[[list[float]], OptimizationResult],
    #              init: OptimizationParameter) -> OptimizationResult:
    #     ...

    def minimize(
        self,
        func: Callable[[list[float]], OptimizationResult],
        init: OptimizationParameter
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
        return self.minimize_(func, init)
        # if init is None:
        #     return self.minimize_(func, None)

        # _init = np.copy(init).flatten()

        # result = self.minimize_(func, _init)
        # return result.fix_dims(cast(tuple[int], init.shape))

    # def check_bounds(self, init: NDArray | None) -> None:
    #     """
    #     Check if the bounds are correctly set. This method should be
    #     called before the optimization starts.
    #     """
    #     if not hasattr(self, "bounds"):
    #         raise OptimizerError("This optimizer requires bounds.")

    #     bounds = getattr(self, "bounds")
    #     if isinstance(bounds, list):
    #         # warnings.warn("WARNING: bounds should be a numpy array. "
    #         #               "Converting to numpy array.")
    #         setattr(self, "bounds", np.array(bounds))
    #         bounds = getattr(self, "bounds")

    #     if bounds.shape[-1] != 2:
    #         raise OptimizerError("Bounds should be a 2D "
    #                              "array with shape (n, 2).")

    #     if init is not None and bounds.shape[:-1] != init.shape:
    #         raise OptimizerError(
    #             f"Bounds shape {bounds.shape[:-1]} "
    #             f"does not match init shape {init.shape}."
    #         )

    @abstractmethod
    def minimize_(
        self,
        func: Callable[[list[float]], OptimizationResult],
        init: OptimizationParameter
    ) -> OptimizationResult:
        """
        Abstract method that should be implemented by the subclass.
        """
