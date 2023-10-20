import numpy as np
import sympy
from sympy.core.expr import Expr
import sys

# dla [0.1e-7]*5, [0.1e-7]*5] po 50 krokach 0.00001 QNG wychodzi 
#'angles': [[ 3.01802149e-5, -2.96978888e-5,  3.63643112e-5,
 #        -5.16468548e-5,  3.49334697e-5],
#        [-1.56217877e+3, -8.12462911e+2, -7.48954477e+2,
 #        -3.69992396e+2, -3.36382973e+2]]
 #po kolejnych 50
 #[[ 2.90348277e-05 -3.64824242e-05  3.46917217e-05 -5.43960632e-05
#   4.48874418e-05]
# [-1.56210092e+03 -8.12399321e+02 -7.48907278e+02 -3.70022896e+02
 # -3.36433279e+02]]
 #49   -5834.00059342468 
 
#010010001, 0.006, 28.0 !!!!

 # po kolejnych 7 kroków  po 0.00005
 #[[ 2.96529440e-05 -4.55437245e-05  3.37496506e-05 -6.31195925e-05
 #  4.83633630e-05]
 #[-1.56203575e+03 -8.12349384e+02 -7.48893468e+02 -3.70059336e+02
 # -3.36474434e+02]]
 # -5872.691880258778 
 
 # kolejne 50 krokow po 0.00002
 # 5.06680543e-05, -7.78639543e-05,  2.19620781e-05,
  #       -1.07823656e-04,  4.83135712e-05],
  #      [-1.56199367e+03, -8.12255548e+02, -7.48868418e+02,
  #       -3.70142551e+02, -3.36542642e+02]
 #-6051.449005813
 
 # kolejne 50 kroków po 0.00003
# [[ 6.32993528e-05, -8.86137488e-05,  2.55910170e-05,
  #       -1.48442690e-04,  4.33636820e-05],
  #      [-1.56191307e+03, -8.12274500e+02, -7.48885446e+02,
   #      -3.70175938e+02, -3.36523168e+02]]
 # -6101.28141583793
 #kolejne 50 kroków po 0.00004
#[[ 7.82393240e-05, -1.06532685e-04,  3.07228711e-05,
 #        -2.04090429e-04,  3.50652006e-05],
 #       [-1.56181706e+03, -8.12321247e+02, -7.48882825e+02,
 #        -3.70198523e+02, -3.36499142e+02]]
# -6166

#dla [0.1e-3]*5, [0.5]*5]30+50+50 QNG przy kroku 0.00001
# wtedy wychodzi:
#Best params: {'angles': tensor([[ 8.31146019e-05,  1.57348810e-04,  1.34276443e-06,
  #        1.23770996e-04, -2.58713482e-05],
 #       [ 8.97996738e-01,  5.73017878e-01,  5.97425600e-01,
 #         6.27467574e-01,  6.78818295e-01]]
 #srednia z H: -5849.607816442087, dalej schodzi już b. powoli 
#import os
#print(os.getcwd())
#sys.path.append("C:\\Users\\kzaja\\Documents\\mariusz\\qhyper\\QHyper\\")
#Best params: {'angles': tensor([[ 8.31146019e-05,  1.57348810e-04,  1.34276443e-06,
  #        1.23770996e-04, -2.58713482e-05],
 #       [ 8.97996738e-01,  5.73017878e-01,  5.97425600e-01,
 #         6.27467574e-01,  6.78818295e-01]]
 # -5849.607816442087 
sys.path.append(".")


hyper_params = {'cost_function_weight': 1.0, # weight for: cost function 
                'encoding_machine_1_weight': 37.89186033670198, # weight for: (x[0] + x[1] + x[2] - 1)**2
                'encoding_machine_2_weight': 37.89186033670198, # weight for: (x[3] + x[4] + x[5] - 1)**2
                'encoding_machine_3_weight': 37.89186033670198, # weight for: (x[6] + x[7] + x[8] - 1)**2
                'deadline_linear_form_weight': 15.536726433137282, # weight for: deadline constraint - linear form (-- this is from the unbalanced penalization approach)
                'deadline_quadratic_form_weight': 38.61604208771982} # weight for: deadline constraint - quadratic form

import numpy as np
import sympy
from sympy.core.expr import Expr

from QHyper.problems.base import Problem
from QHyper.util import Expression

deadline = 13
class SimpleWorkflowProblem(Problem):

    
    def __init__(self) -> None:
        num_of_qubits = 9 #wsp.workflow.cost_matrix.shape[0] * wsp.workflow.cost_matrix.shape[1]
        self.variables = sympy.symbols(' '.join([f'x{i}' for i in range(num_of_qubits)]))                                  
        self._set_objective_function()
        self._set_constraints()
        
    def _set_objective_function(self) -> None:
        C_f = 6.0*self.variables[0] + 8.0*self.variables[1] + 8.0*self.variables[2] + 3.0*self.variables[3] + 4.0*self.variables[4] + 4.0*self.variables[5] + 12.0*self.variables[6] + 16.0*self.variables[7] + 16.0*self.variables[8]
        

        K_f4_linear = deadline - (6*self.variables[0] + 2*self.variables[1] + 4*self.variables[2] + 3*self.variables[3] +
                            1*self.variables[4] + 2*self.variables[5] + 12*self.variables[6] + 4*self.variables[7] + 8*self.variables[8])
                
        self.objective_function = Expression(hyper_params['cost_function_weight'] * C_f + hyper_params['deadline_linear_form_weight'] *  K_f4_linear)
        
    def _set_constraints(self):
        K_f1 = self.variables[0] + self.variables[1] + self.variables[2] - 1
        K_f2 = self.variables[3] + self.variables[4] + self.variables[5] - 1
        K_f3 = self.variables[6] + self.variables[7] + self.variables[8] - 1

        K_f4_squared = deadline - (6*self.variables[0] + 2*self.variables[1] + 4*self.variables[2] + 3*self.variables[3] +
                            1*self.variables[4] + 2*self.variables[5] + 12*self.variables[6] + 4*self.variables[7] + 8*self.variables[8])

            
        self.constraints = [Expression(K_f1), Expression(K_f2), Expression(K_f3), Expression(K_f4_squared)]
    
    def get_score(self, result, penalty=0):
        
        x = [int(val) for val in result]
    
        if (x[0] + x[1] + x[2] == 1 and 
            x[3] + x[4] + x[5] == 1 and 
            x[6] + x[7] + x[8] == 1 and 
            6*x[0] + 2*x[1] + 4*x[2] + 3*x[3] + 1*x[4] + 2*x[5] + 12*x[6] + 4*x[7] + 8*x[8] <= 13):
            
            return 6.0*x[0] + 8.0*x[1] + 8.0*x[2] + 3.0*x[3] + 4.0*x[4] + 4.0*x[5] + 12.0*x[6] + 16.0*x[7] + 16.0*x[8]
        
        return penalty
    
problem = SimpleWorkflowProblem()
print(f"Variables used to describe objective function"
      f" and constraints: {problem.variables}")
print(f"Objective function: {problem.objective_function}")
print("Constraints (RHS == 0):")
for constraint in problem.constraints:
    print(f"    {constraint}")
    
params_cofing = {
         'angles':  [[ 6.32993528e-05, -8.86137488e-05,  2.55910170e-05,
         -1.48442690e-04,  4.33636820e-05],
        [-1.56191307e+03, -8.12274500e+02, -7.48885446e+02,
         -3.70175938e+02, -3.36523168e+02]],
       # 'angles': [[0.1e-7]*5, [0.1e-7]*5], # QAOA angles - first we have gammas (for the cost Hamiltonian), then we have betas (for the mixer)
        'hyper_args': [1, # do not change - this should be the weight for the 'cost function' but since in our cost function 
                          # we also have the deadline in the linear form (as of now it needs to be implemented this way due to QHyper limitations)
                          # the weight for the actual cost function is set there. THIS WILL NOT WORK WELL WITH HYPER-QAOA.
                          
                       hyper_params['encoding_machine_1_weight'], 
                       hyper_params['encoding_machine_2_weight'], 
                       hyper_params['encoding_machine_3_weight'], 
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
best_params = vqa.evaluate(params_cofing)
print(f"Best params: {best_params}")

best_results = tester.evaluate(best_params, print_results=True)
print(f"Best results: {best_results}")
print(f"Params used for optimizer:\n{best_params['angles']},\n"
      f"and params used for hyperoptimizer: {best_params['hyper_args']}")
