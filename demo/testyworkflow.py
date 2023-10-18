import numpy as np
import sympy
from sympy.core.expr import Expr
import sys
#import os
#print(os.getcwd())
#sys.path.append("C:\\Users\\kzaja\\Documents\\mariusz\\qhyper\\QHyper\\")
sys.path.append(".")


from QHyper.problems.workflow_scheduling import (
    Workflow,
    WorkflowSchedulingProblem,
)

tasks_file =  ".\\demo\\workflows_data\\workflows\\3_tasks_3_machines_1_path.json"
machines_file = ".\\demo\\workflows_data\\machines\\machines_for_3_tasks_3_machines_1_path.json"
deadline = 13

workflow = Workflow(tasks_file, machines_file, deadline)
wsp = WorkflowSchedulingProblem(workflow)