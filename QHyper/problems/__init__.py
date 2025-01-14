# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
Any problem that is in directory 'QHyper/custom' or 'custom' will be
automatically imported and available for use. Add 'name' attribute to the
class to make it available in the PROBLEMS dictionary (if not problem will be
available by its class name).

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
"""

from typing import Type, Any
import copy

from QHyper.util import search_for

from QHyper.problems.base import Problem


custom_fetched = False
custom_problems = {}

def getProblem(name: str) -> Type[Problem]:
    """Get problem class by name. Used for creating Problem objects from config.
    """
    global custom_fetched
    global custom_problems

    if not custom_fetched:
        custom_problems = (search_for(Problem, 'QHyper/custom')
                                    | search_for(Problem, 'custom'))
        custom_fetched = True

    name_ = name.lower()

    if name_ in custom_problems:
        return custom_problems[name_]
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


def problem_from_config(config: dict[str, dict[str, Any]]) -> Problem:
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
    problem_class = getProblem(problem_type)

    return problem_class(**config_)
