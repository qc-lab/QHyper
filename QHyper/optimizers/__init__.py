from typing import Type

from .base import Optimizer, OptimizationResult  # noqa: F401

from .cem import CEM
from .qml_gradient_descent import QmlGradientDescent
from .random import Random
from .scipy_minimizer import ScipyOptimizer
from .basinhopping import Basinhopping
from .grid_search import GridSearch
from .dummy import Dummy

OPTIMIZERS_BY_NAME: dict[str, Type[Optimizer]] = {
    'scipy': ScipyOptimizer,
    'random': Random,
    'qml': QmlGradientDescent,
    'cem': CEM,
    'basinhopping': Basinhopping,
    'grid': GridSearch,
    'dummy': Dummy,
}
