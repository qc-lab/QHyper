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
import pkgutil
import importlib

from typing import Type, Any

from QHyper.problems import problem_from_config, ProblemConfigException
from QHyper.util import search_for

from QHyper.optimizers import Optimizer, create_optimizer, OptimizationParameter

from QHyper.solvers.base import (  # noqa F401
    Solver, SolverResult, SolverConfigException)
from QHyper.solvers.hyper_optimizer import HyperOptimizer

# from .vqa.base import VQA
# from .classical.gurobi import Gurobi
# from .quantum_annealing.cqm import CQM
# from .quantum_annealing.dqm import DQM
# from .quantum_annealing.advantage import Advantage
#

# SOLVERS: dict[str, Type[Solver]] = {
#     'vqa': VQA,
#     'gurobi': Gurobi,
#     'cqm': CQM,
#     'dqm': DQM,
#     'advantage': Advantage
# }
# SOLVERS.update(search_for(Solver, 'QHyper/custom'))
# SOLVERS.update(search_for(Solver, 'custom'))


def get_solver(name: str, category: str, platform: str) -> Solver:
    try:
        module = importlib.import_module(
            f'QHyper.solvers.{category}.{platform}.{name.lower()}')
        if hasattr(module, name) and issubclass(getattr(module, name), Solver):
            return getattr(module, name)
        raise ImportError
    except ImportError:
        raise FileNotFoundError(
            "Solver with"
            f"{f'category \'{category}\' and ' if category else ''}"
            f"{f'platform \'{platform}\' and ' if platform else ''}"
            f"{name} not found"
        )
    # import pathlib
    # current_path = pathlib.Path(os.path.relpath(__file__))

    # search_path = f'{category}/{platform}/*.py'
    # print(current_path.parent)
    # print(str(current_path.parent.joinpath(search_path)))
    # file_names = glob.glob(str(
    #     current_path.parent.joinpath(search_path).resolve()))
    # print(file_names)
    # if len(file_names) == 0:
    #     raise FileNotFoundError(
    #         "Solver with"
    #         f"{f'category \'{category}\' and ' if category else ''}"
    #         f"{f'platform \'{platform}\' and ' if platform else ''}"
    #         f"{name} not found"
    #     )

    # for file_name in file_names:
    #     print(file_name)
    #     try:
    #         module = importlib.import_module(file_name)
    #         print(module)

    #         if hasattr(module, name) and issubclass(getattr(module, name),
    #                                                 Solver):
    #             return getattr(module, name)
    #     except ImportError as e:
    #         print(e)
    #         continue
    # raise FileNotFoundError(
    #     "Solver with"
    #     f"{f'category \'{category}\' and ' if category else ''}"
    #     f"{f'platform \'{platform}\' and ' if platform else ''}"
    #     f"{name} not found"
    # )


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
    if 'category' not in config['solver']:
        raise SolverConfigException("Solver category was not provided")
    if 'platform' not in config['solver']:
        raise SolverConfigException("Solver platform was not provided")

    try:
        solver_class = get_solver(
            config['solver']['name'],
            config['solver']['category'],
            config['solver']['platform']
        )
    except FileNotFoundError:
        raise SolverConfigException(
            f"Solver {config['solver']['name']} not found"
        )
    import dataclasses
    # print(dataclasses.fields(solver_class))

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
            # if config['solver'][field.name].get('type') in OPTIMIZERS:
            #     optimizer_class = OPTIMIZERS[
            #         config['solver'][field.name].pop('type')]
            # else:
            #     raise SolverConfigException(
            #         f"Optimizer {config['solver'][field.name].get('type')} "
            #         "not found"
            #     )

            # config['solver'][field.name] = optimizer_class(
            #     **config['solver'][field.name]
            # )

    try:
        error_msg = "Solver configuration was not provided"
        solver_config = config.pop('solver')
        error_msg = "Solver name was not provided"
        solver_name = solver_config.pop('name')
        error_msg = "Solver category was not provided"
        solver_category = solver_config.pop('category')
        error_msg = "Solver platform was not provided"
        solver_platform = solver_config.pop('platform')
        solver_class = get_solver(
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
