from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from wfcommons import Instance
from wfcommons.utils import read_json


@dataclass
class TargetMachine:
    name: str
    memory: int
    cpu: dict[str, float]
    price: float
    memory_cost_multiplier: float


def main():
    instance = Instance(
        "/Users/jzawalska/Coding/QHyper/QHyper/problems/workflows_data/workflows/msc_sample_workflow.json")
    # "/Users/jzawalska/Coding/QHyper/QHyper/problems/workflows_data/workflows/generated.json")

    workflow = instance.workflow
    tasks = workflow.nodes(data=True)
    deadline = 32

    all_paths = []
    for root in instance.roots():
        for leaf in instance.leaves():
            paths = nx.all_simple_paths(instance.workflow, root, leaf)
            all_paths.extend(paths)

    # new_machines = read_json("/Users/jzawalska/Coding/QHyper/QHyper/problems/workflows_data/machines/cyfronet.json")
    new_machines = read_json("/Users/jzawalska/Coding/QHyper/QHyper/problems/workflows_data/machines/msc_sample_machines.json")

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
            number_of_operations = task["task"].runtime * task_machine.cpu_speed * task_machine.cpu_cores * 10**6
            real_runtime = number_of_operations / (machine_details.cpu["speed"] * machine_details.cpu["count"] * 10**6)
            # print(f'machine_name {machine_name}, task_name {name}, task_runtime {task["task"].runtime}, number_of_operations {number_of_operations}, real_runtime {real_runtime}')
            machine_runtime.append(real_runtime)
            machine_cost.append(real_runtime * machine_details.price)
        costs[machine_name] = machine_cost
        runtimes[machine_name] = machine_runtime
    #

    cost_df = pd.DataFrame(data=costs, index=instance.workflow.nodes)
    runtime_df = pd.DataFrame(data=runtimes, index=instance.workflow.nodes)

    # print("\nCost matrix")
    # print(cost_df)
    # print("Time matrix")
    # print(runtime_df)

    # print(cost_df["ZeusCpu"]["merge_ID0000022"])
    # print(cost_df.iloc[2,1]) # [task, machine]
    # print(cost_df.columns[3]) #machine : columns[modulo num of machines]
    # print(cost_df.index[0]) #task : index[// division num machines]
    columns = cost_df.columns
    rows = cost_df.index.to_numpy()
    print(columns[0])
    # print(cost_df.index)


if __name__ == "__main__":
    main()

