# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
===============================
Solvers (:mod:`QHyper.solvers`)
===============================

.. currentmodule:: QHyper.solvers

Any solver that is in directory 'QHyper/custom' or 'custom' will be
automatically imported and available for use. Add 'name' attribute to the
class to make it available in the SOLVERS dictionary (if not solver will be
available by its class name).

Interfaces
==========

.. autosummary::
    :toctree: generated/

    Solver  -- Base class for solvers.
    SolverResult -- Dataclass for storing


Classical Solvers
=================

.. autosummary::
    :toctree: generated/

    Gurobi -- Gurobi solver.


Quantum Annealing Solvers
=========================

.. autosummary::
    :toctree: generated/

    CQM -- CQM solver.
    DQM -- DQM solver.
    Advantage -- Advantage solver.


Variational Quantum Algorithm Solvers
=====================================

.. autosummary::
    :toctree: generated/
    :recursive:

    vqa -- VQA solver.
"""

import copy

from typing import Type, Any

from QHyper.problems import problem_from_config, ProblemConfigException
from QHyper.util import search_for

from QHyper.solvers.base import (  # noqa F401
    Solver, SolverResult, SolverConfigException)

from .vqa.base import VQA
from .classical.gurobi import Gurobi
from .quantum_annealing.cqm import CQM
from .quantum_annealing.dqm import DQM
from .quantum_annealing.advantage import Advantage


SOLVERS: dict[str, Type[Solver]] = {
    'vqa': VQA,
    'gurobi': Gurobi,
    'cqm': CQM,
    'dqm': DQM,
    'advantage': Advantage
}
SOLVERS.update(search_for(Solver, 'QHyper/custom'))
SOLVERS.update(search_for(Solver, 'custom'))


def solver_from_config(config: dict[str, Any]) -> Solver:
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
    try:
        error_msg = "Solver configuration was not provided"
        solver_config = config.pop('solver')
        error_msg = "Solver type was not provided"
        solver_type = solver_config.pop('type')
        error_msg = f"There is no {solver_type} solver type"
        solver_class = SOLVERS[solver_type]
    except KeyError:
        raise SolverConfigException(error_msg)

    solver = solver_class.from_config(problem, solver_config)

    return solver
