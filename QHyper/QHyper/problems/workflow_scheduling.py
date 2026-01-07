# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import math
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import pandas as pd
import sympy
from wfcommons import Instance
from wfcommons.utils import read_json

from networkx.classes.reportviews import NodeView

from sympy.core.expr import Expr
from typing import cast

from QHyper.constraint import Constraint, Operator, UNBALANCED_PENALIZATION
from QHyper.parser import from_sympy
from QHyper.polynomial import Polynomial
from QHyper.problems.base import Problem, ProblemException


@dataclass
class TargetMachine:
    name: str
    memory: int
    cpu: dict[str, float]
    price: float
    memory_cost_multiplier: float


class Workflow:
    def __init__(self, tasks_file: str, machines_file: str, deadline: float) -> None:
        self.wf_instance = Instance(tasks_file)
        self.tasks = self._get_tasks()
        self.machines = self._get_machines(machines_file)
        self.deadline = deadline
        self._set_paths()
        self.time_matrix, self.cost_matrix = self._calc_dataframes()
        self.task_names = self.time_matrix.index
        self.machine_names = self.time_matrix.columns

    def _get_tasks(self) -> NodeView:
        return self.wf_instance.workflow.nodes(data=True)

    def _get_machines(self, machines_file: str) -> dict[str, TargetMachine]:
        target_machines = read_json(machines_file)
        return {
            machine["name"]: TargetMachine(**machine)
            for machine in target_machines["machines"]
        }

    def _set_paths(self) -> None:
        all_paths = []
        for root in self.wf_instance.roots():
            for leaf in self.wf_instance.leaves():
                paths = nx.all_simple_paths(self.wf_instance.workflow, root, leaf)
                all_paths.extend(paths)

        self.paths = all_paths

    def _calc_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        costs, runtimes = {}, {}
        for machine_name, machine_details in self.machines.items():
            machine_cost, machine_runtime = [], []
            for task_name, task in self.tasks:
                old_machine = task["task"].machine
                number_of_operations = (
                    task["task"].runtime * old_machine.cpu_speed * old_machine.cpu_cores
                )
                # todo can this overflow?
                real_runtime = number_of_operations / (
                    machine_details.cpu["speed"] * machine_details.cpu["count"]
                )
                machine_runtime.append(real_runtime)
                machine_cost.append(real_runtime * machine_details.price)
            costs[machine_name] = machine_cost
            runtimes[machine_name] = machine_runtime

        time_df = pd.DataFrame(data=runtimes, index=self.wf_instance.workflow.nodes)
        cost_df = pd.DataFrame(data=costs, index=self.wf_instance.workflow.nodes)

        return time_df, cost_df


def calc_slack_coefficients(constant: int) -> list[int]:
    num_slack = int(math.floor(math.log2(constant)))
    slack_coefficients = [2**j for j in range(num_slack)]
    if constant - 2**num_slack >= 0:
        slack_coefficients.append(constant - 2**num_slack + 1)
    return slack_coefficients


class WorkflowSchedulingProblem(Problem):
    """Workflow Scheduling Problem

    Parameters
    ----------
    encoding : str
        Encoding used for the problem (one-hot or binary)
    tasks_file : str
        Path to the tasks file
    machines_file : str
        Path to the machines file
    deadline : float
        Deadline for the workflow

    Attributes
    ----------
    objective_function: Polynomial
        Objective_function represented as a Polynomial
    constraints : list[Polynomial]
        List of constraints represented as a Polynomials
    """

    def __new__(
        cls, encoding: str, tasks_file: str, machines_file: str, deadline: float
    ) -> 'WorkflowSchedulingOneHot | WorkflowSchedulingBinary':
        workflow = Workflow(tasks_file, machines_file, deadline)

        if encoding == "one-hot":
            return WorkflowSchedulingOneHot(workflow)
        elif encoding == "binary":
            return WorkflowSchedulingBinary(workflow)
        raise ProblemException(f"Unsupported encoding: {encoding}")


class WorkflowSchedulingOneHot(Problem):
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.variables: tuple[sympy.Symbol] = sympy.symbols(" ".join([
            f"x{i}" for i in range(
                len(self.workflow.tasks) * len(self.workflow.machines)
            )
        ]))
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

        self.objective_function = from_sympy(expression)

    def _set_constraints(self) -> None:
        self.constraints: list[Constraint] = []

        # machine assignment constraint
        for task_id in range(len(self.workflow.time_matrix.index)):
            expression: Expr = cast(Expr, 0)
            for machine_id in range(len(self.workflow.time_matrix.columns)):
                expression += self.variables[
                    machine_id + task_id * len(self.workflow.time_matrix.columns)
                ]
            self.constraints.append(
                Constraint(from_sympy(expression), Polynomial(1)))

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
                    from_sympy(expression),
                    Polynomial(self.workflow.deadline),
                    Operator.LE,
                    UNBALANCED_PENALIZATION,
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


class WorkflowSchedulingBinary(Problem):
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
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
                    self.workflow.cost_matrix.loc[task_name, machine_name]
                    * current_term
                )

        self.objective_function = from_sympy(expression)

    def _set_constraints(self) -> None:
        self.constraints: list[Constraint] = []

        for path in self.workflow.paths:
            expression: sympy.Expr = sympy.Expr(0)
            for _, task_name in enumerate(path):
                for _, machine_name in enumerate(self.workflow.time_matrix.columns):
                    current_term = cast(Expr, 1)
                    task_id = self.workflow.time_matrix.index.get_loc(task_name)
                    assert isinstance(task_id, int)

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
                    from_sympy(expression),
                    Polynomial(self.workflow.deadline),
                    Operator.LE,
                    UNBALANCED_PENALIZATION,
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
