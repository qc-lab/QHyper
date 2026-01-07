# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
This module contains implementaions of different problems.
Problems are defined using :py:class:`~QHyper.polynomial.Polynomial` module.
No problem is imported by deafult to reduce number of dependencies.
To use any problem you can import it directly like

.. code-block:: python

    from QHyper.problems.knapsack import KnapsackProblem

or use function :py:func:`Problems.get` with the name of the problem.
Any problem that is in directory 'QHyper/custom' or 'custom' will be
also available in this function.

.. rubric:: Interface

.. autosummary::
    :toctree: generated

    Problem  -- Base class for problems.

.. rubric:: Available problems

.. autosummary::
    :toctree: generated

    knapsack.KnapsackProblem -- Knapsack problem.
    tsp.TravelingSalesmanProblem -- Traveling Salesman Problem.
    maxcut.MaxCutProblem -- Max-Cut problem.
    workflow_scheduling.WorkflowSchedulingProblem -- Workflow Scheduling problem.
    community_detection.CommunityDetectionProblem -- Community Detection problem.

.. rubric:: Additional functions

.. autoclass:: Problems
    :members:

"""

from typing import Type, Any
import copy

from QHyper.util import search_for

from QHyper.problems.base import Problem


class Problems:
    custom_problems: None | dict[str, type] = None

    @staticmethod
    def get(name: str) -> Type[Problem]:
        """
        Get problem class by name. Used for creating Problem objects from config.

        The problem will be available by the 'name' attribute if defined or
        by the class name. Letters case doesn't matter.
        """

        if Problems.custom_problems is None:
            Problems.custom_problems = (
                search_for(Problem, 'QHyper/custom')
                | search_for(Problem, 'custom'))

        name_ = name.lower()

        if name_ in Problems.custom_problems:
            return Problems.custom_problems[name_]
        elif name_ in ["knapsack", "knapsackproblem"]:
            from .knapsack import KnapsackProblem
            return KnapsackProblem
        elif name_ in ["tsp", "travelingsalesmanproblem"]:
            from .tsp import TravelingSalesmanProblem
            return TravelingSalesmanProblem
        elif name_ in ["maxcut", "maxcutproblem"]:
            from .maxcut import MaxCutProblem
            return MaxCutProblem
        elif name_ in ["workflow_scheduling", "workflowschedulingproblem"]:
            from .workflow_scheduling import WorkflowSchedulingProblem
            return WorkflowSchedulingProblem
        elif name_ in ["community_detection", "communitydetectionproblem"]:
            from .community_detection import CommunityDetectionProblem
            return CommunityDetectionProblem
        else:
            raise ValueError(f"Problem {name} not found")


class ProblemConfigException(Exception):
    """Exception raised when problem configuration is incorrect"""
    pass


def problem_from_config(config: dict[str, Any]) -> Problem:
    """
    Create Problem object from provided configuration.

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
