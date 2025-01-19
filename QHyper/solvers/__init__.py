# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
This module contains implementations of different solvers.
In QHyper exists three types of solvers: classical, quantum annealing and
gate-based.
Some of them are written from scratch based on popular algorithms, while
others are just a wrapper for existing solutions.
No solver is imported by deafult to reduce number of dependencies.
To use any solver you can import it directly like

.. code-block:: python

    from QHyper.solver.gate_based.pennylane.qaoa import QAOA

or use function :py:func:`Solvers.get` with the name, category, and platform.
Any solver that is in directory 'QHyper/custom' or 'custom' will be
also available in this function.

.. rubric:: Interface

.. autosummary::
    :toctree: generated/

    Solver  -- Base class for solvers.
    SolverResult -- Dataclass for storing results


.. rubric:: Classical Solvers

.. autosummary::
    :toctree: generated/

    classical.gurobi.Gurobi -- Gurobi solver.


.. rubric:: Quantum Annealing Solvers

.. autosummary::
    :toctree: generated/

    quantum_annealing.dwave.cqm.CQM -- CQM solver.
    quantum_annealing.dwave.dqm.DQM -- DQM solver.
    quantum_annealing.dwave.advantage.Advantage -- Advantage solver.


.. rubric:: Gate-based solvers

.. autosummary::
    :toctree: generated/

    gate_based.pennylane.qaoa.QAOA -- QAOA solver.
    gate_based.pennylane.qml_qaoa.QML_QAOA -- QML QAOA solver.
    gate_based.pennylane.wf_qaoa.WF_QAOA -- Weight Free QAOA solver.
    gate_based.pennylane.h_qaoa.H_QAOA -- Hyper QAOA solver.


.. rubric:: Hyper-optimizer

Not really a solver, but a class that can be used to optimize the hyperparameters
of another solver. It is a wrapper around the solver class.

.. autosummary::
    :toctree: generated/

    hyper_optimizer.HyperOptimizer -- Hyper-optimizer.

.. rubric:: Additional functions

.. autoclass:: Solvers
    :members:
"""

import copy
import dataclasses


from typing import Type, Any

from QHyper.problems import problem_from_config, ProblemConfigException
from QHyper.util import search_for

from QHyper.optimizers import Optimizer, create_optimizer, OptimizationParameter

from QHyper.solvers.base import (  # noqa F401
    Solver, SolverResult, SolverConfigException)
from QHyper.solvers.hyper_optimizer import HyperOptimizer


class Solvers:
    custom_solvers: None | dict[str, type] = None

    @staticmethod
    def get(name: str, category: str = '', platform: str = '') -> Type[Solver]:
        """
        Get solver class by name, category, and platform.

        The name is required, other paramters might be required
        if there would be more than one solver with the same name.
        The solver will be available by the 'name' attribute if defined or
        by the class name. Letters case doesn't matter.

        """

        if Solvers.custom_solvers is None:
            Solvers.custom_solvers = (
                search_for(Solver, 'QHyper/custom')
                | search_for(Solver, 'custom'))

        # In the future, the category and platform might be required for some
        # solvers
        if category == "custom":
            if name in Solvers.custom_solvers:
                return Solvers.custom_solvers[name]
            else:
                raise FileNotFoundError(
                    f"Solver {name} not found in custom solvers"
                )

        name_ = name.lower()
        if name_ in ["qaoa"]:
            from .gate_based.pennylane.qaoa import QAOA
            return QAOA
        elif name_ in ["qml_qaoa"]:
            from .gate_based.pennylane.qml_qaoa import QML_QAOA
            return QML_QAOA
        elif name_ in ["wf_qaoa"]:
            from .gate_based.pennylane.wf_qaoa import WF_QAOA
            return WF_QAOA
        elif name_ in ["h_qaoa"]:
            from .gate_based.pennylane.h_qaoa import H_QAOA
            return H_QAOA
        elif name_ in ["gurobi"]:
            from .classical.gurobi.gurobi import Gurobi
            return Gurobi
        elif name_ in ["cqm"]:
            from .quantum_annealing.dwave.cqm import CQM
            return CQM
        elif name_ in ["dqm"]:
            from .quantum_annealing.dwave.dqm import DQM
            return DQM
        elif name_ in ["advantage"]:
            from .quantum_annealing.dwave.advantage import Advantage
            return Advantage
        else:
            raise SolverConfigException(f"Solver {name} not found")


def solver_from_config(config: dict[str, Any]) -> Solver | HyperOptimizer:
    """
    Alternative way of creating solver.
    Expect dict with two keys:
    - type - type of solver
    - args - arguments which will be passed to Solver instance

    Parameters
    ----------
    problem : Problem
        The problem to be solved
    config : dict[str. Any]
        Configuration in form of dict

    Returns
    -------
    Solver
        Initialized Solver object
    """

    config = copy.deepcopy(config)

    try:
        problem_config = config.pop('problem')
    except KeyError:
        raise ProblemConfigException(
            'Problem configuration was not provided')
    problem = problem_from_config(problem_config)

    error_msg = ""

    if 'solver' not in config:
        raise SolverConfigException("Solver configuration was not provided")
    if 'name' not in config['solver']:
        raise SolverConfigException("Solver name was not provided")

    try:
        solver_class = Solvers.get(
            config['solver']['name'],
            config['solver'].get('category', ''),
            config['solver'].get('platform', '')
        )
    except FileNotFoundError:
        raise SolverConfigException(
            f"Solver {config['solver']['name']} not found"
        )

    for field in dataclasses.fields(solver_class):
        if not field.init:
            continue

        if field.name not in config['solver']:
            continue

        if field.type == Optimizer:
            config['solver'][field.name] = create_optimizer(
                config['solver'][field.name]
            )
        elif field.type == OptimizationParameter:
            config['solver'][field.name] = OptimizationParameter(
                **config['solver'][field.name]
            )

    try:
        error_msg = "Solver configuration was not provided"
        solver_config = config.pop('solver')
        error_msg = "Solver name was not provided"
        solver_name = solver_config.pop('name')
        solver_category = solver_config.pop('category', '')
        solver_platform = solver_config.pop('platform', '')
        solver_class = Solvers.get(
            solver_name, solver_category, solver_platform)
    except KeyError:
        raise SolverConfigException(error_msg)

    solver = solver_class.from_config(problem, solver_config)

    hyper_optimizer_config = config.pop('hyper_optimizer', None)

    if hyper_optimizer_config:
        optimizer_config = hyper_optimizer_config.pop('optimizer')
        optimizer = create_optimizer(optimizer_config)
        hyper_optimizer = HyperOptimizer(
            optimizer, solver, **hyper_optimizer_config)
        return hyper_optimizer
    return solver
