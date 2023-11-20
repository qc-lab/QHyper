import math
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import pandas as pd
import sympy
import math
from wfcommons import Instance
from wfcommons.utils import read_json

from networkx.classes.reportviews import NodeView

from sympy.core.expr import Expr
from typing import cast

from QHyper.util import Expression, Constraint, MethodsForInequalities, Operator
from .base import Problem
from .workflow_scheduling import Workflow


def calc_slack_coefficients(constant: int) -> list[int]:
    num_slack = int(math.floor(math.log2(constant)))
    slack_coefficients = [2**j for j in range(num_slack)]
    if constant - 2**num_slack >= 0:
        slack_coefficients.append(constant - 2**num_slack + 1)
    return slack_coefficients


class WorkflowSchedulingBinary(Problem):
    def __init__(self, workflow: Workflow):
        self.workflow: Workflow = workflow
        self.variables: tuple[sympy.Symbol] = sympy.symbols(
            " ".join(
                [
                    f"x{i}"
                    for i in range(
                        len(self.workflow.tasks)
                        * math.ceil(math.log2(len(self.workflow.machines)))
                    )
                ]
            )
        )
        self._set_binary_representation()
        self._set_objective_function()
        self._set_constraints()

    def _set_binary_representation(self) -> None:
        num_of_machines = len(self.workflow.machines)
        len_machine_encoding = math.ceil(math.log2(num_of_machines))

        self.machines_binary_representation = {
            machine_name: bin(machine_id)[2:].zfill(len_machine_encoding)
            for machine_name, machine_id in zip(
                self.workflow.machine_names, range(len(self.workflow.machines))
            )
        }

    def _set_objective_function(self) -> None:
        expression = cast(Expr, 0)
        for _, task_name in enumerate(self.workflow.time_matrix.index):
            for _, machine_name in enumerate(self.workflow.time_matrix.columns):
                current_term = cast(Expr, 1)
                task_id = self.workflow.time_matrix.index.get_loc(task_name)
                variable_id = task_id * (len(self.workflow.tasks) - 1)
                for el in self.machines_binary_representation[machine_name]:
                    if el == "0":
                        current_term *= 1 - self.variables[variable_id]
                    elif el == "1":
                        current_term *= self.variables[variable_id]
                    variable_id += 1
                expression += (
                    self.wsp.cost_matrix.loc[task_name, machine_name] * current_term
                )

        self.objective_function: Expression = Expression(expression)

    def _set_constraints(self) -> None:
        self.constraints: list[Constraint] = []

        # deadline constraint
        for path in self.workflow.paths:
            expression = self.calculate_expression_with_coefficients(
                self.workflow.time_matrix
            )

            for path in self.workflow.paths:
                expression = 0
                for _, task_name in enumerate(path):
                    for _, machine_name in enumerate(self.workflow.time_matrix.columns):
                        current_term = cast(Expr, 1)
                        task_id = self.workflow.time_matrix.index.get_loc(task_name)
                        variable_id = task_id * (len(self.workflow.tasks) - 1)
                        for el in self.machines_binary_representation[machine_name]:
                            if el == "0":
                                current_term *= 1 - self.variables[variable_id]
                            elif el == "1":
                                current_term *= self.variables[variable_id]
                            variable_id += 1
                        expression += (
                            self.workflow.time_matrix.loc[task_name, machine_name]
                            * current_term
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

    def get_score(self, result: str, penalty: float = 0) -> float:
        decoded_solution = {}
        machine_encoding_len = math.ceil(math.log2(len(self.workflow.machines)))
        for task_id, task_name in enumerate(self.workflow.task_names):
            decoded_solution[task_name] = int(
                result[
                    task_id * machine_encoding_len : task_id * machine_encoding_len
                    + machine_encoding_len
                ],
                2,
            )

        for path in self.workflow.paths:
            path_time = 0
            for task_name in path:
                machine_name = self.workflow.time_matrix.columns[
                    decoded_solution[task_name]
                ]
                path_time += self.workflow.time_matrix.loc[task_name, machine_name]

            if path_time > self.workflow.deadline:
                return penalty

        cost_of_used_machines = 0
        for task_id, task_name in enumerate(self.workflow.task_names):
            machine_name = self.workflow.time_matrix.columns[
                decoded_solution[task_name]
            ]
            cost_of_used_machines += self.workflow.cost_matrix.loc[
                task_name, machine_name
            ]

        return cost_of_used_machines
