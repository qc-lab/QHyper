import numpy as np
import sympy
from sympy.core.expr import Expr
import sys


sys.path.append(".")

#waga machine 37.89186033670198
#machine_weight=37.89186033670198
machine_weight=20
hyper_params = {'cost_function_weight': 1, # weight for: cost function 
                'encoding_machine_1_weight': machine_weight, # weight for: (x[0] + x[1] + x[2] - 1)**2
                'encoding_machine_2_weight': machine_weight, # weight for: (x[3] + x[4] + x[5] - 1)**2
                'encoding_machine_3_weight': machine_weight, # weight for: (x[6] + x[7] + x[8] - 1)**2
                'deadline_linear_form_weight': 1, # weight for: deadline constraint - linear form (-- this is from the unbalanced penalization approach)
                'deadline_quadratic_form_weight': 2} # weight for: deadline constraint - quadratic form

import numpy as np
import sympy
from sympy.core.expr import Expr

from QHyper.problems.base import Problem
from QHyper.util import Expression

deadline = 13
class SimpleWorkflowProblem(Problem):

    
    def __init__(self) -> None:
        num_of_qubits = 6 #wsp.workflow.cost_matrix.shape[0] * wsp.workflow.cost_matrix.shape[1]
        self.variables = sympy.symbols(' '.join([f'x{i}' for i in range(num_of_qubits)]))                                  
        self._set_objective_function()
        self._set_constraints()
        
    def _set_objective_function(self) -> None:
        
        C_f = (6.0*(1-self.variables[0])*(1-self.variables[1]) 
        + 8.0*(1-self.variables[0])*(self.variables[1])  
        + 8.0*(self.variables[0])*(1-self.variables[1]) 
        + 2.0*(self.variables[0])*(self.variables[1])  
        + 3.0*(1-self.variables[2])*(1-self.variables[3]) 
        + 4.0*(1-self.variables[2])*(self.variables[3])  
        + 4.0*(self.variables[2])*(1-self.variables[3]) 
        + 1.0*(self.variables[2])*(self.variables[3])
        + 12.0*(1-self.variables[4])*(1-self.variables[5]) 
        + 16.0*(1-self.variables[4])*(self.variables[5])  
        + 16.0*(self.variables[4])*(1-self.variables[5]) 
        + 4.0*(self.variables[4])*(self.variables[5])  )
        
        self.deadline_aux=(6.0*(1-self.variables[0])*(1-self.variables[1]) 
       # + 2.0*(1-self.variables[0])*(self.variables[1])  
      #  + 4.0*(self.variables[0])*(1-self.variables[1]) 
       # + 16.0*(self.variables[0])*(self.variables[1])  
        + 3.0*(1-self.variables[2])*(1-self.variables[3]) 
        #+ 1.0*(1-self.variables[2])*(self.variables[3])  
       # + 2.0*(self.variables[2])*(1-self.variables[3]) 
       # + 8.0*(self.variables[2])*(self.variables[3])
        + 12.0*(1-self.variables[4])*(1-self.variables[5]) )
       # + 4.0*(1-self.variables[4])*(self.variables[5])  
       # + 8.0*(self.variables[4])*(1-self.variables[5]) 
       # + 32.0*(self.variables[4])*(self.variables[5])  )

        K_f4_linear = deadline - self.deadline_aux
        K_f4_squared=K_f4_linear**2
        K_f4_squared1=sympy.expand(K_f4_squared)

        print(K_f4_squared1, "\n\n")
          
        for i in range(6):
            K_f4_squared1=K_f4_squared1.subs(self.variables[i]**2, self.variables[i])
            print(K_f4_squared1,"\n\n")  
        
        print(K_f4_squared1)     
        self.objective_function = (Expression(hyper_params['cost_function_weight'] * C_f 
                                              + hyper_params['deadline_linear_form_weight'] *  K_f4_linear)
                                   + hyper_params['deadline_quadratic_form_weight'] * K_f4_linear**2)
        
    def _set_constraints(self):
       # print(self.deadline_aux)

        
      
            
        self.constraints = []
    
    def get_score(self, result, penalty=0):
        
        x = [int(val) for val in result]
    
        if (6.0*(1.0-x[0])*(1-x[1]) 
        + 2.0*(1.0-x[0])*(x[1])  
        + 4.0*(x[0])*(1.0-x[1]) 
        + 16.0*(x[0])*(x[1])  
        + 3.0*(1.0-x[2])*(1.0-x[3]) 
        + 1.0*(1.0-x[2])*(x[3])  
        + 2.0*(x[2])*(1.0-x[3]) 
        + 8.0*(x[2])*(x[3])
        + 12.0*(1.0-x[4])*(1.0-x[5]) 
        + 4.0*(1.0-x[4])*(x[5])  
        + 8.0*(x[4])*(1.0-x[5]) 
        + 32.0*(x[4])*(x[5]) <=13 ):
            
            return (6.0*(1.0-x[0])*(1.0-x[1]) 
        + 8.0*(1.0-x[0])*(x[1])  
        + 8.0*(x[0])*(1.0-x[1]) 
        + 2.0*(x[0])*(x[1])  
        + 3.0*(1.0-x[2])*(1.0-x[3]) 
        + 4.0*(1.0-x[2])*(x[3])  
        + 4.0*(x[2])*(1.0-x[3]) 
        + 1.0*(x[2])*(x[3])
        + 12.0*(1.0-x[4])*(1.0-x[5]) 
        + 16.0*(1.0-x[4])*(x[5])  
        + 16.0*(x[4])*(1.0-x[5]) 
        + 4.0*(x[4])*(x[5]) ) 
        
        return penalty
    
problem = SimpleWorkflowProblem()
print(f"Variables used to describe objective function"
      f" and constraints: {problem.variables}")
print(f"Objective function: {problem.objective_function}")
print("Constraints (RHS == 0):")
for constraint in problem.constraints:
    print(f"    {constraint}")
    
params_cofing = {
      #  'angles': [[-1.46546167e-03, -1.47630043e-03,  5.23969216e-04,
     #    4.32957649e-04,  7.74771138e-04],
      #  [-5.95883641e+01, -8.44669790e-01, -8.87181353e-01,
      #   -5.11625786e-01, -5.02607625e-01]],

        'angles': [[0.1e-13]*5, [np.pi/2]*5], # QAOA angles - first we have gammas (for the cost Hamiltonian), then we have betas (for the mixer)
        'hyper_args': [1, # do not change - this should be the weight for the 'cost function' but since in our cost function 
                          # we also have the deadline in the linear form (as of now it needs to be implemented this way due to QHyper limitations)
                          # the weight for the actual cost function is set there. THIS WILL NOT WORK WELL WITH HYPER-QAOA.
                          
                   
                       hyper_params['deadline_quadratic_form_weight']],
    }
# Simple quantum circuit without optimzers will be used to test the results
# WF-QAOA is choosen becasue this PQC has most suitable evaluation function
from QHyper.solvers.vqa.base import VQA
tester_config = {
    'pqc': {
        'type': 'wfqaoa',
        'layers': 5,
    }
}

tester = VQA(problem, config=tester_config)
solver_config2 = {
    'pqc': {
        'type': 'qaoa',
        'layers': 5,
        'mixer': 'pl_x_mixer',
    },
     'optimizer': {
        'type': 'scipy',
        'maxfun': 600,
    },
}
solver_config = {
    'pqc': {
        'type': 'sqaoa',
        'layers': 5
    }
}
vqa = VQA(problem, config=solver_config)
# dla QNG trzeba użyć evaluate(), a nie solve() bo tak jest zaszyte w sqaoa
#best_params = vqa.evaluate(params_cofing)
best_params = vqa.evaluate(params_cofing)
print(f"Best params: {best_params}")

best_results = tester.evaluate(best_params, print_results=True)
print(f"Best results: {best_results}")
print(f"Params used for optimizer:\n{best_params['angles']},\n"
      f"and params used for hyperoptimizer: {best_params['hyper_args']}")
