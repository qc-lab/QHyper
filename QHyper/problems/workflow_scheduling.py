import networkx as nx
import pandas as pd
import sympy
from wfcommons import Instance
from wfcommons.utils import read_json

from QHyper.problems.integrate_wfcommons import TargetMachine


class Workflow:
    def __init__(self):

        instance = Instance(
            "./workflows_data/workflows/msc_sample_workflow.json")

        workflow = instance.workflow
        tasks = workflow.nodes(data=True)

        all_paths = []
        for root in instance.roots():
            for leaf in instance.leaves():
                paths = nx.all_simple_paths(instance.workflow, root, leaf)
                all_paths.extend(paths)

        self.paths = all_paths

        new_machines = read_json(
            "./workflows_data/machines/msc_sample_machines.json")

        target_machines = {
            machine['name']: TargetMachine(**machine)
            for machine in new_machines["machines"]
        }

        # calc dataframe
        costs, runtimes = {}, {}
        for machine_name, machine_details in target_machines.items():
            machine_cost = []
            machine_runtime = []
            for name, task in tasks:
                task_machine = task["task"].machine
                number_of_operations = task["task"].runtime * task_machine.cpu_speed * task_machine.cpu_cores * 10 ** 6
                real_runtime = number_of_operations / (
                        machine_details.cpu["speed"] * machine_details.cpu["count"] * 10 ** 6)
                machine_runtime.append(real_runtime)
                machine_cost.append(real_runtime * machine_details.price)
            costs[machine_name] = machine_cost
            runtimes[machine_name] = machine_runtime
        #

        self.time_matrix = pd.DataFrame(data=costs, index=instance.workflow.nodes)
        self.cost_matrix = pd.DataFrame(data=runtimes, index=instance.workflow.nodes)

        # todo check if it is safe to use index-columns from dataframe <- make sure of this when you create the dataframe
        self.tasks = self.time_matrix.index
        self.machines = self.time_matrix.columns
        self.deadline = 32
        self.x = sympy.symbols(
            ' '.join([f'x{i}' for i in range(len(self.tasks) * len(self.machines))]))  # num of bin variables


class WorkflowSchedulingProblem:
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self._create_objective_function()
        self._create_constraints()

    def _create_objective_function(self) -> None:
        equation = 0

        for task_id, task_name in enumerate(self.workflow.tasks):
            for machine_id, machine_name in enumerate(self.workflow.machines):
                cost = self.workflow.cost_matrix[machine_name][task_name]
                equation += cost * self.workflow.x[machine_id + task_id * len(self.workflow.machines)]

        self.objective_function = equation

    def _create_constraints(self) -> None:  # one w sumie nie musza byc kwadratowe
        constraints = []

        # constraint one_machine
        for task_id in range(len(self.workflow.tasks)):
            equation = 0
            for machine_id in range(len(self.workflow.machines)):
                equation += self.workflow.x[machine_id + task_id * len(self.workflow.machines)]
            equation -= 1
            constraints.append(sympy.Rel(equation, 0, "=="))

        # constraint deadline #todo ok
        for path in self.workflow.paths:
            equation = 0
            for task_id, task_name in enumerate(self.workflow.tasks):
                for machine_id, machine_name in enumerate(self.workflow.machines):
                    if task_name in path:
                        time = self.workflow.time_matrix[machine_name][task_name]
                        equation += time * self.workflow.x[machine_id + task_id * len(self.workflow.machines)]

            equation -= self.workflow.deadline
            constraints.append(sympy.Rel(equation, 0, "<="))

        self.constraints = constraints


if __name__ == "__main__":
    workflow = Workflow()
    wsp = WorkflowSchedulingProblem(workflow)
    print(wsp.objective_function)
    print(wsp.constraints)
