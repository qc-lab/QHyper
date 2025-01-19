# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
This module contains implementations of different optimizers.
Some of the optimizers are written from scratch based on popular algorithms,
while others are just a wrapper for existing solutions.
No optimizer is imported by deafult to reduce number of dependencies.
To use any optimizer you can import it directly like

.. code-block:: python

    from QHyper.optimizers.random import Random

or use function :py:func:`Optimizers.get` with the name of the optimizer.
Any optimizer that is in directory 'QHyper/custom' or 'custom' will be
also available in this function.

.. rubric:: Optimization result dataclass

.. autosummary::
    :toctree: generated

    OptimizationResult
    OptimizationParameter

.. rubric:: Interface

.. autosummary::
    :toctree: generated

    Optimizer

.. rubric:: Available optimizers

.. autosummary::
    :toctree: generated

    scipy_minimizer.ScipyOptimizer -- Wrapper for the scipy.optimize.minimize function.
    qml_gradient_descent.QmlGradientDescent -- Wrapper for the PennyLane gradient descent optimizers.
    cem.CEM -- Cross-entropy method optimizer.
    random.Random -- Random search optimizer.
    grid_search.GridSearch -- Grid search optimizer.
    dummy.Dummy -- Dummy optimizer.

.. rubric:: Additional functions

.. autoclass:: Optimizers
    :members:

"""
import copy
from typing import Type, Any

from QHyper.util import search_for

from QHyper.optimizers.base import (                # noqa: F401
    Optimizer, OptimizationResult, OptimizerError, OptimizationParameter)  # noqa: F401

from .dummy import Dummy


class Optimizers:
    custom_optimizers: None | dict[str, type] = None

    @staticmethod
    def get(name: str) -> Type[Optimizer]:
        """
        Get Optimizer class by name.

        The optimizer will be available by the 'name' attribute if defined or
        by the class name. Letters case doesn't matter.
        """

        if Optimizers.custom_optimizers is None:
            Optimizers.custom_optimizers = (
                search_for(Optimizer, 'QHyper/custom')
                | search_for(Optimizer, 'custom'))

        name_ = name.lower()

        if name_ in Optimizers.custom_optimizers:
            return Optimizers.custom_optimizers[name_]
        elif name_ in ["scipy", "scipyminimizer"]:
            from .scipy_minimizer import ScipyOptimizer
            return ScipyOptimizer
        elif name_ in ["random", "randomsearch"]:
            from .random import Random
            return Random
        elif name_ in ["qml", "qmlgradientdescent"]:
            from .qml_gradient_descent import QmlGradientDescent
            return QmlGradientDescent
        elif name_ in ["cem", "crossentropymethod"]:
            from .cem import CEM
            return CEM
        elif name_ in ["grid", "gridsearch"]:
            from .grid_search import GridSearch
            return GridSearch
        elif name_ in ["dummy"]:
            return Dummy
        else:
            raise OptimizerError(f"Optimizer {name} not found")


def create_optimizer(config: dict[str, Any]) -> Optimizer:
    config_ = copy.deepcopy(config)
    opt_type = config_.pop('type')

    optimizer_class = Optimizers.get(opt_type)
    return optimizer_class(**config_)
