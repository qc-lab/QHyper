from typing import Type
import copy

from .knapsack import KnapsackProblem
from .tsp import TSPProblem
from .maxcut import MaxCutProblem
from .workflow_scheduling import WorkflowSchedulingProblem

from .base import Problem


PROBLEMS_BY_NAME: dict[str, Type[Problem]] = {
    'knapsack': KnapsackProblem,
    'tsp': TSPProblem,
    'maxcut': MaxCutProblem,
    'workflow_scheduling': WorkflowSchedulingProblem,
}


class ProblemConfigException(Exception):
    pass


def problem_from_config(config: dict[str, dict[str, any]]) -> Problem:
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
    _config = copy.deepcopy(config)
    try:
        error_msg = "Problem configuration was not provided"
        problem_type = _config.pop('type')
        error_msg = f"There is no {problem_type} problem type"
        problem_class = PROBLEMS_BY_NAME[problem_type]
    except KeyError:
        raise ProblemConfigException(error_msg)

    return problem_class(**_config)
