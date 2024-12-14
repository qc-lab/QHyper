# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
=================================
Problems (:mod:`QHyper.problems`)
=================================

.. currentmodule:: QHyper.problems

Any problem that is in directory 'QHyper/custom' or 'custom' will be
automatically imported and available for use. Add 'name' attribute to the
class to make it available in the PROBLEMS dictionary (if not problem will be
available by its class name).

Package Content
===============

.. autosummary::
    :toctree: generated/

    Problem  -- Base class for problems.

    KnapsackProblem -- Knapsack problem.
    TSPProblem -- Traveling Salesman Problem.
    MaxCutProblem -- Max-Cut problem.
    WorkflowSchedulingProblem -- Workflow Scheduling problem.
    CommunityDetectionProblem -- Community Detection problem.

"""

from typing import Type, Any
import copy

# from .knapsack import KnapsackProblem
# from .tsp import TSPProblem
# from .maxcut import MaxCutProblem
# from .workflow_scheduling import WorkflowSchedulingProblem
# from .community_detection import CommunityDetectionProblem, Network # noqa 401

from QHyper.util import search_for

from QHyper.problems.base import Problem


class Problems:
    custom_fetched = False
    custom_problems = {}

    @staticmethod
    def get(name: str) -> Type[Problem]:
        if not Problems.custom_fetched:
            Problems.custom_problems = (search_for(Problem, 'QHyper/custom')
                                        | search_for(Problem, 'custom'))
            Problems.custom_fetched = True

        if name in Problems.custom_problems:
            return Problems.custom_problems[name]
        elif name == "knapsack":
            from .knapsack import KnapsackProblem
            return KnapsackProblem
        elif name == "tsp":
            from .tsp import TSPProblem
            return TSPProblem
        elif name == "maxcut":
            from .maxcut import MaxCutProblem
            return MaxCutProblem
        elif name == "workflow_scheduling":
            from .workflow_scheduling import WorkflowSchedulingProblem
            return WorkflowSchedulingProblem
        elif name == "community_detection":
            from .community_detection import CommunityDetectionProblem
            return CommunityDetectionProblem
        else:
            raise ValueError(f"Problem {name} not found")

# PROBLEMS.update(search_for(Problem, 'QHyper/custom'))
# PROBLEMS.update(search_for(Problem, 'custom'))


class ProblemConfigException(Exception):
    pass


def problem_from_config(config: dict[str, dict[str, Any]]) -> Problem:
    """
    Parameters
    ----------
    config : dict[str. Any]
        Configuration in form of dict

    Returns
    -------
    Problem
        Initialized Problem object
    """
    config_ = copy.deepcopy(config)
    if "type" not in config_:
        raise ProblemConfigException("Problem type was not provided")
    problem_type = config_.pop('type')
    problem_class = Problems.get(problem_type)

    return problem_class(**config_)
