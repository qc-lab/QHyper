from typing import Type

from .base import Optimizer

from .cem import CEM
from .qml_gradient_descent import QmlGradientDescent
from .random import Random
from .scipy_minimizer import ScipyOptimizer
# from .basinhopping import Basinhopping
from ._basinhopping import Basinhopping

OPTIMIZERS_BY_NAME: dict[str, Type[Optimizer]] = {
    'scipy': ScipyOptimizer,
    'random': Random,
    'qml': QmlGradientDescent,
    'cem': CEM,
    'basinhopping': Basinhopping
}
