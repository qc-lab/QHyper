from QHyper.problems.workflow_scheduling import (
    Workflow,
    WorkflowSchedulingProblem,
)

tasks_file =  "/Users/jzawalska/Coding/QHyper-vsc/QHyper/demo/workflows_data/workflows/3_tasks_3_machines_1_path.json"
machines_file = "/Users/jzawalska/Coding/QHyper-vsc/QHyper/demo/workflows_data/machines/machines_for_3_tasks_3_machines_1_path.json"
deadline = 13

workflow = Workflow(tasks_file, machines_file, deadline)
wsp = WorkflowSchedulingProblem(workflow)

for constraint in wsp.constraints:
    print(constraint)

# print(wsp.workflow.time_matrix)    
# print(wsp.workflow.time_matrix.iloc[0, 0])
# print(wsp.workflow.time_matrix.iloc[0, 1])
# print(wsp.workflow.time_matrix.iloc[1, 0]) #task, # machine




def test_get_score():
    result = "010100001" 
    # result = "011100001" 
    # result = "010100011" 


    x = [int(val) for val in result]
    
    num_machines = len(workflow.machines)
    for i in range(num_machines):
        
        print(sum(x[i * num_machines:(i+1) * num_machines]))
        if sum(x[i * num_machines:(i+1) * num_machines]) != 1:
            assert False
    assert True
    
    #decode values from slacks
    
    
    


# def decode_solution(self, solution: dict) -> dict:
#         decoded_solution = {}
#         for variable, value in solution.items():
#             _, id = variable[0], int(variable[1:])  # todo add validation
#             if value == 1.0:
#                 machine_id = id % len(self.workflow.machines)
#                 task_id = id // len(self.workflow.machines)
#                 decoded_solution[self.workflow.time_matrix.index[task_id]] = (
#                     self.workflow.time_matrix.columns[machine_id])

#         return decoded_solution