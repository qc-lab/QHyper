# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
=====================================
Optimizers (:mod:`QHyper.optimizers`)
=====================================

.. currentmodule:: QHyper.optimizers

Any optimizer that is in directory 'QHyper/custom' or 'custom' will be
automatically imported and available for use. Add 'name' attribute to the
class to make it available in the OPTIMIZERS dictionary (if not optimizer will
be available by its class name).

Package Content
===============

.. autosummary::
    :toctree: generated/

    Optimizer  -- Base class for optimizers.
    OptimizationResult -- Dataclass for storing the results of an optimization run.

    ScipyOptimizer -- Wrapper for the scipy.optimize.minimize function.
    QmlGradientDescent -- Wrapper for the PennyLane gradient descent optimizers.
    CEM -- Cross-entropy method optimizer.
    Random -- Random search optimizer.
    GridSearch -- Grid search optimizer.
    Dummy -- Dummy optimizer.

"""
import copy
from typing import Type, Any

from QHyper.util import search_for

from QHyper.optimizers.base import (                # noqa: F401
    Optimizer, OptimizationResult, OptimizerError, OptimizationParameter)  # noqa: F401

# from .cem import CEM
# from .qml_gradient_descent import QmlGradientDescent
# from .random import Random
# from .scipy_minimizer import ScipyOptimizer
# # from .basinhopping import Basinhopping
# from .grid_search import GridSearch
from .dummy import Dummy


class Optimizers:
    custom_fetched = False
    custom_optimizers = {}

    @staticmethod
    def get(name: str) -> Type[Optimizer]:
        if not Optimizers.custom_fetched:
            Optimizers.custom_optimizers = (
                search_for(Optimizer, 'QHyper/custom')
                | search_for(Optimizer, 'custom'))
            Optimizers.custom_fetched = True

        if name in Optimizers.custom_optimizers:
            return Optimizers.custom_optimizers[name]
        elif name.lower() == "scipy" or name.lower() == "scipyminimizer":
            from .scipy_minimizer import ScipyOptimizer
            return ScipyOptimizer
        elif name.lower() == "random":
            from .random import Random
            return Random
        elif name.lower() == "qml" or name.lower() == "qmlgradientdescent":
            from .qml_gradient_descent import QmlGradientDescent
            return QmlGradientDescent
        elif name.lower() == "cem":
            from .cem import CEM
            return CEM
        elif name.lower() == "grid" or name.lower() == "gridsearch":
            from .grid_search import GridSearch
            return GridSearch
        elif name.lower() == "dummy":
            return Dummy
        else:
            raise OptimizerError(f"Optimizer {name} not found")


# OPTIMIZERS: dict[str, Type[Optimizer]] = {
#     'scipy': ScipyOptimizer,
#     'random': Random,
#     'qml': QmlGradientDescent,
#     'cem': CEM,
#     # 'basinhopping': Basinhopping,
#     'grid': GridSearch,
#     'dummy': Dummy,
# }
# OPTIMIZERS.update(search_for(Optimizer, 'QHyper/custom'))
# OPTIMIZERS.update(search_for(Optimizer, 'custom'))


def create_optimizer(config: dict[str, Any]) -> Optimizer:
    config_ = copy.deepcopy(config)
    opt_type = config_.pop('type')

    optimizer_class = Optimizers.get(opt_type)
    return optimizer_class(**config_)
