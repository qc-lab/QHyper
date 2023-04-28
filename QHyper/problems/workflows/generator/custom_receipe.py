#type: ignore
from logging import Logger
import random
from typing import Optional, Set, List
import pathlib

from wfcommons.common import Task, File, TaskType, Workflow
from wfcommons.common.machine import Machine
from wfcommons.wfchef.wfchef_abstract_recipe import BaseMethod, WfChefWorkflowRecipe
import networkx as nx

this_dir = pathlib.Path(__file__).resolve().parent


class CustomRecipe(WfChefWorkflowRecipe):

    def __init__(self,
                 data_footprint: Optional[int] = 0,
                 num_tasks: Optional[int] = 3,
                 exclude_graphs: Set[str] = set(),
                 runtime_factor: Optional[float] = 1.0,
                 input_file_size_factor: Optional[float] = 1.0,
                 output_file_size_factor: Optional[float] = 1.0,
                 logger: Optional[Logger] = None,
                 base_method: BaseMethod = BaseMethod.ERROR_TABLE,
                 **kwargs) -> None:
        super().__init__(
            name="Customdata",
            data_footprint=data_footprint,
            num_tasks=num_tasks,
            exclude_graphs=exclude_graphs,
            runtime_factor=runtime_factor,
            input_file_size_factor=input_file_size_factor,
            output_file_size_factor=output_file_size_factor,
            logger=logger,
            this_dir=this_dir,
            base_method=base_method,
            **kwargs
        )

    def generate_nx_graph(self, G) -> nx.DiGraph:
        attributes = {}
        for node in G.nodes():
            attributes[node] = {'type': TaskType.COMPUTE, 'id': str(node)}

        G.add_nodes_from(['SRC', 'DST'])
        attributes['SRC'] = {'type': TaskType.COMPUTE, 'id': 'SRC'}
        attributes['DST'] = {'type': TaskType.COMPUTE, 'id': 'DST'}

        nx.set_node_attributes(G, attributes)
        return G

    def _generate_task(self, task_name: str, task_id: str) -> Task:
        runtime: float = random.randint(1, 10)  # todo

        self.tasks_files[task_id] = []
        self.tasks_files_names[task_id] = []
        task = Task(
            name=task_id,  # task_name
            task_id=task_id,  # '0{}'.format(task_id.split('_0')[1]),
            task_type=TaskType.COMPUTE,
            runtime=runtime,
            machine=Machine("default_machine", {
                "count": 1,
                "speed": 1,
            }),
            args=[],
            cores=1,
            avg_cpu=None,
            bytes_read=None,
            bytes_written=None,
            memory=random.randint(20000, 40000), #todo
            energy=None,
            avg_power=None,
            priority=None,
            files=[]
        )
        self.tasks_map[task_id] = task
        return task

    def _generate_task_files(self, task: Task) -> List[File]:
        return []

    def build_workflow(self, G, workflow_name: Optional[str] = None) -> Workflow:
        """Generate a synthetic workflow instance.

        :param workflow_name: The workflow name
        :type workflow_name: int

        :return: A synthetic workflow instance object.
        :rtype: Workflow
        """
        workflow = Workflow(name=self.name + "-synthetic-instance" if not workflow_name else workflow_name,
                            makespan=0)
        print("G.nodes()", G.nodes())
        graph = self.generate_nx_graph(G)
        print("graph.nodes()", graph.nodes())
        task_names = {}
        for node in graph.nodes:
            print(graph.nodes[node])
            if node in ["SRC", "DST"]:
                continue
            node_type = graph.nodes[node]["type"]
            task_name = self._generate_task_name(node_type)
            task = self._generate_task(node_type, task_name)
            workflow.add_node(task_name, task=task)

            task_names[node] = task_name

        # tasks dependencies
        for (src, dst) in graph.edges:
            if src in ["SRC", "DST"] or dst in ["SRC", "DST"]:
                continue
            workflow.add_edge(task_names[src], task_names[dst])

            if task_names[src] not in self.tasks_children:
                self.tasks_children[task_names[src]] = []
            if task_names[dst] not in self.tasks_parents:
                self.tasks_parents[task_names[dst]] = []

            self.tasks_children[task_names[src]].append(task_names[dst])
            self.tasks_parents[task_names[dst]].append(task_names[src])

        # find leaf tasks
        leaf_tasks = []
        for node_name in workflow.nodes:
            task: Task = workflow.nodes[node_name]['task']
            if task.name not in self.tasks_children:
                leaf_tasks.append(task)

        for task in leaf_tasks:
            self._generate_task_files(task)

        workflow.nxgraph = graph
        self.workflows.append(workflow)
        return workflow
