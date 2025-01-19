# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import abc
import dataclasses

from abc import abstractmethod

from typing import Callable, Self


@dataclasses.dataclass
class OptimizationParameter:
    """
    Dataclass for storing bounds, steps and init values for parameters
    that might be optimized. Most of the time some of this values are not
    required, but it depends on the chosen optimization algorithm.
    Check the documentation of the chosen algorithm to see which values are
    required.

    Attributes
    ----------
    min : list[float]
        List of minimum values for each parameter.
    max : list[float]
        List of maximum values for each parameter.
    step : list[float]
        List of step values for each parameter. Used for example in the grid
        search algorithm. For 0-th parameter the following values will be
        generated: min[0], min[0] + step[0], min[0] + 2*step[0], ...
    init : list[float]
        List of initial values for each parameter. Some algorithms require
        starting point to be set.
    """

    min: list[float] = dataclasses.field(default_factory=list)
    max: list[float] = dataclasses.field(default_factory=list)
    step: list[float] = dataclasses.field(default_factory=list)
    init: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.min = list(self.min)
        self.max = list(self.max)
        self.step = list(self.step)
        self.init = list(self.init)

    def assert_bounds(self) -> None:
        """Check if bounds are correctly set.
        """

        if not self.min:
            raise ValueError("Min bounds are required")
        if not self.max:
            raise ValueError("Max bounds are required")
        if len(self.min) != len(self.max):
            raise ValueError("Min and Max bounds must have the same length")

    def assert_step(self) -> None:
        """Check if steps are correctly set.
        """
        self.assert_bounds()
        if len(self.step) == 0:
            raise ValueError("Steps are required")
        if len(self.min) != len(self.step):
            raise ValueError("Steps must have the same length as bounds")

    def assert_init(self) -> None:
        """Check if init values are correctly set.
        """
        if len(self.init) == 0:
            raise ValueError("Init are required")

    def assert_bounds_init(self) -> None:
        """Check if bounds and init values are correctly set.
        """
        self.assert_bounds()
        self.assert_init()
        if len(self.min) != len(self.init):
            raise ValueError("Init must have the same length as bounds")

    def __add__(self, other: Self) -> 'OptimizationParameter':
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
               init: list[float] | None = None) -> 'OptimizationParameter':
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
    def bounds(self) -> list[tuple[float, float]]:
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
    params : list[float]
        The optimal point (function arguments) found by the optimization
        algorithm.
    history : list[list[OptimizationResult]]
        The history of the optimization algorithm. Each element of the list
        represents the results of the objective function at each
        iteration - there can be multiple results per each iteration (epoch).
    """

    value: float
    params: list[float]
    history: list[list['OptimizationResult']] = dataclasses.field(
        default_factory=list)


class Optimizer(abc.ABC):
    """
    Base class for Optimizer.

    """

    @abstractmethod
    def minimize(
        self,
        func: Callable[[list[float]], OptimizationResult],
        init: OptimizationParameter
    ) -> OptimizationResult:
        """
        Method that minimizes the given function using the
        implemented optimization algorithm.
        This method has to be implemented by the subclass.

        Parameters
        ----------
        func : Callable[[list[float]], OptimizationResult]
            The objective function to be minimized.
        init : OptimizationParameter
            The initial parameter for the optimization algorithm.
            The required fields are defined by subclass.

        Returns
        -------
        OptimizationResult
            Result contains the minimum function value, the
            corresponding optimal point, and the history of
            the optimization.
        """
