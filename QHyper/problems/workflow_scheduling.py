from dataclasses import dataclass

import networkx as nx
import pandas as pd
import sympy
from wfcommons import Instance
from wfcommons.utils import read_json

from QHyper.hyperparameter_gen.parser import Expression
from QHyper.problems.problem import Problem


@dataclass
class TargetMachine:
    name: str
    memory: int
    cpu: dict[str, float]
    price: float
    memory_cost_multiplier: float


class Workflow:
    def __init__(self, tasks_file, machines_file, deadline):
        self.wf_instance = Instance(tasks_file)
        self.tasks = self._get_tasks(tasks_file)
        self.machines = self._get_machines(machines_file)
        self.deadline = deadline
        self._set_paths()
        self.time_matrix, self.cost_matrix = self._calc_dataframes()
        # self.task_names = self.time_matrix.index
        # self.machine_names = self.time_matrix.columns

    def _get_tasks(self, tasks_file):
        return self.wf_instance.workflow.nodes(data=True)

    def _get_machines(self, machines_file):
        target_machines = read_json(machines_file)
        return {machine['name']: TargetMachine(**machine) for machine in target_machines["machines"]}

    def _set_paths(self):
        all_paths = []
        for root in self.wf_instance.roots():
            for leaf in self.wf_instance.leaves():
                paths = nx.all_simple_paths(self.wf_instance.workflow, root, leaf)
                all_paths.extend(paths)

        self.paths = all_paths

    def _calc_dataframes(self):
        costs, runtimes = {}, {}
        for machine_name, machine_details in self.machines.items():
            machine_cost, machine_runtime = [], []
            for task_name, task in self.tasks:
                old_machine = task["task"].machine
                number_of_operations = task["task"].runtime * old_machine.cpu_speed * old_machine.cpu_cores * 10 ** 6
                real_runtime = number_of_operations / (  # todo can this overflow?
                        machine_details.cpu["speed"] * machine_details.cpu["count"] * 10 ** 6)
                machine_runtime.append(real_runtime)
                machine_cost.append(real_runtime * machine_details.price)
            costs[machine_name] = machine_cost
            runtimes[machine_name] = machine_runtime

        time_df = pd.DataFrame(data=runtimes, index=self.wf_instance.workflow.nodes)
        cost_df = pd.DataFrame(data=costs, index=self.wf_instance.workflow.nodes)

        return time_df, cost_df


class WorkflowSchedulingProblem(Problem):
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.variables = sympy.symbols(
            ' '.join([f'x{i}' for i in range(len(self.workflow.tasks) * len(self.workflow.machines))]))
        self._set_objective_function()
        self._set_constraints()

    def _set_objective_function(self) -> None:
        expression = 0
        for task_id, task_name in enumerate(self.workflow.time_matrix.index):
            for machine_id, machine_name in enumerate(self.workflow.time_matrix.columns):
                cost = self.workflow.cost_matrix[machine_name][task_name]
                expression += cost * self.variables[machine_id + task_id * len(self.workflow.time_matrix.columns)]

        self.objective_function = Polynomial(expression)

    def _set_constraints(self) -> None:
        constraints = {"==": [],
                       "<=": [],
                       ">=": []}

        # machine assignment constraint
        for task_id in range(len(self.workflow.time_matrix.index)):
            expression = 0
            for machine_id in range(len(self.workflow.time_matrix.columns)):
                expression += self.variables[machine_id + task_id * len(self.workflow.time_matrix.columns)]
            expression -= 1

            constraints["=="].append(Polynomial(expression))

        # deadline constraint
        for path in self.workflow.paths:
            expression = 0
            for task_id, task_name in enumerate(self.workflow.time_matrix.index):
                for machine_id, machine_name in enumerate(self.workflow.time_matrix.columns):
                    if task_name in path:
                        time = self.workflow.time_matrix[machine_name][task_name]
                        expression += time * self.variables[machine_id + task_id * len(self.workflow.time_matrix.columns)]

            expression -= self.workflow.deadline
            constraints["<="].append(Polynomial(expression))

        self.constraints = constraints

    def decode_solution(self, solution):
        decoded_solution = {}
        for variable, value in solution.items():
            name, id = variable[0], int(variable[1:])  # todo add validation
            if value == 1.0:
                machine_id = id % len(self.workflow.machines)
                task_id = id // len(self.workflow.machines)
                decoded_solution[self.workflow.time_matrix.index[task_id]] = self.workflow.time_matrix.columns[
                    machine_id]

        return decoded_solution
