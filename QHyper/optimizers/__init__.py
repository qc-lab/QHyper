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

from typing import Type

from QHyper.util import search_for

from QHyper.optimizers.base import Optimizer, OptimizationResult  # noqa: F401

from .cem import CEM
from .qml_gradient_descent import QmlGradientDescent
from .random import Random
from .scipy_minimizer import ScipyOptimizer
# from .basinhopping import Basinhopping
from .grid_search import GridSearch
from .dummy import Dummy

OPTIMIZERS: dict[str, Type[Optimizer]] = {
    'scipy': ScipyOptimizer,
    'random': Random,
    'qml': QmlGradientDescent,
    'cem': CEM,
    # 'basinhopping': Basinhopping,
    'grid': GridSearch,
    'dummy': Dummy,
}
OPTIMIZERS.update(search_for(Optimizer, 'QHyper/custom'))
OPTIMIZERS.update(search_for(Optimizer, 'custom'))
