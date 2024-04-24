# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Type

from QHyper.util import search_for

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
users_optimizers = search_for(Optimizer, 'QHyper/optimizers')
OPTIMIZERS_BY_NAME.update({
    optimizer.__name__.lower(): optimizer
    for optimizer in users_optimizers
})
