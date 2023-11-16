import math
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import sympy
from math import ceil, log2
from wfcommons import Instance
from wfcommons.utils import read_json

from networkx.classes.reportviews import NodeView

from sympy.core.expr import Expr
from typing import cast

from QHyper.util import Expression, Constraint, MethodsForInequalities, Operator
from .base import Problem
from .workflow_scheduling import Workflow


def calc_slack_coefficients(constant: int) -> list[int]:
    num_slack = int(np.floor(np.log2(constant)))
    slack_coefficients = [2**j for j in range(num_slack)]
    if constant - 2**num_slack >= 0:
        slack_coefficients.append(constant - 2**num_slack + 1)
    return slack_coefficients


def generate_binary_strings(length, current=''):
    if length == 0:
        return [current]
    else:
        strings_with_0 = generate_binary_strings(length - 1, current + '0')
        strings_with_1 = generate_binary_strings(length - 1, current + '1')
        return strings_with_0 + strings_with_1


class WorkflowSchedulingProblem(Problem):
    def __init__(self, workflow: Workflow):
        self.workflow: Workflow = workflow
        self.variables: tuple[sympy.Symbol] = sympy.symbols(
            " ".join(
                [
                    f"x{i}"
                    for i in range(
                        len(self.workflow.tasks) * ceil(log2(len(self.workflow.machines)))
                    )
                ]
            )
        )
        self._set_objective_function()
        self._set_constraints()

    def _set_objective_function(self) -> None:
        expression: Expr = cast(Expr, 0)
        for task_id, task_name in enumerate(self.workflow.time_matrix.index):
            for machine_id, machine_name in enumerate(
                self.workflow.time_matrix.columns
            ):
                cost = self.workflow.cost_matrix[machine_name][task_name]
                expression += (
                    cost
                    * self.variables[
                        machine_id + task_id * len(self.workflow.time_matrix.columns)
                    ]
                )

        self.objective_function: Expression = Expression(expression)

    def _set_constraints(self) -> None:
        self.constraints: list[Constraint] = []

        # machine assignment constraint
        for task_id in range(len(self.workflow.time_matrix.index)):
            expression: Expr = cast(Expr, 0)
            for machine_id in range(len(self.workflow.time_matrix.columns)):
                expression += self.variables[
                    machine_id + task_id * len(self.workflow.time_matrix.columns)
                ]
            self.constraints.append(Constraint(Expression(expression), 1, Operator.EQ))

        # deadline constraint
        for path in self.workflow.paths:
            expression = cast(Expr, 0)
            for task_id, task_name in enumerate(self.workflow.time_matrix.index):
                for machine_id, machine_name in enumerate(
                    self.workflow.time_matrix.columns
                ):
                    if task_name in path:
                        time = self.workflow.time_matrix[machine_name][task_name]
                        expression += (
                            time
                            * self.variables[
                                machine_id
                                + task_id * len(self.workflow.time_matrix.columns)
                            ]
                        )

            # todo add constraints unbalanced penalization
            self.constraints.append(
                Constraint(
                    Expression(expression),
                    self.workflow.deadline,
                    Operator.LE,
                    MethodsForInequalities.UNBALANCED_PENALIZATION,
                )
            )

    def decode_solution(self, solution: dict) -> dict:
        decoded_solution = {}
        for variable, value in solution.items():
            _, id = variable[0], int(variable[1:])  # todo add validation
            if value == 1.0:
                machine_id = id % len(self.workflow.machines)
                task_id = id // len(self.workflow.machines)
                decoded_solution[
                    self.workflow.time_matrix.index[task_id]
                ] = self.workflow.time_matrix.columns[machine_id]

        return decoded_solution

    def get_deadlines(self) -> tuple[float, float]:  # todo test this function
        """Calculates the minimum and maximum path runtime
        for the whole workflow."""

        flat_runtimes = [
            (runtime, name)
            for n, machine_runtimes in self.workflow.time_matrix.items()
            for runtime, name in zip(machine_runtimes, self.workflow.task_names)
        ]

        max_path_runtime = 0.0
        min_path_runtime = 0.0

        for path in self.workflow.paths:
            max_runtime: defaultdict[str, float] = defaultdict(lambda: 0.0)
            min_runtime: defaultdict[str, float] = defaultdict(lambda: math.inf)

            for runtime, name in flat_runtimes:
                if name not in path:
                    continue
                max_runtime[name] = max(max_runtime[name], runtime)
                min_runtime[name] = min(min_runtime[name], runtime)
            max_path_runtime = max(max_path_runtime, sum(max_runtime.values()))
            min_path_runtime = max(min_path_runtime, sum(min_runtime.values()))

        return min_path_runtime, max_path_runtime

    def get_score(self, result: str, penalty: float = 0) -> float:
        x = [int(val) for val in result]

        return penalty
