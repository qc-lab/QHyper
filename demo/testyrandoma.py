import numpy as np
import sympy
from sympy.core.expr import Expr
import sys
#import os
#print(os.getcwd())
#sys.path.append("C:\\Users\\kzaja\\Documents\\mariusz\\qhyper\\QHyper\\")
sys.path.append(".")


from QHyper.problems.base import Problem
from QHyper.util import Expression
    
def calc_slack_coefficients(constant: int) -> list[int]:
    num_slack = int(np.floor(np.log2(constant)))
    slack_coefficients = [2 ** j for j in range(num_slack)]
    if constant - 2 ** num_slack >= 0:
        slack_coefficients.append(constant - 2 ** num_slack + 1)
    return slack_coefficients

class SimpleProblem(Problem):
    
    def __init__(self) -> None:
        self.slack_coefficients = calc_slack_coefficients(5)
        self.variables = sympy.symbols(' '.join([f'x{i}' for i in range(2)])  + ' ' 
                                    + ' '.join([f'x{i+2}' for i in range(len(self.slack_coefficients))]))
        self._set_objective_function()
        self._set_constraints()
        
    def _set_objective_function(self) -> None:
        C_f = 2 * self.variables[0] + 5 * self.variables[1] + self.variables[0] * self.variables[1]
        self.objective_function = Expression(C_f)
        
    def _set_constraints(self):
        K_f1 = self.variables[0] + self.variables[1] - 1
        
        K_f2 = 5 * self.variables[0] + 2 * self.variables[1] 
        for i, coefficient in enumerate(self.slack_coefficients):
           K_f2 += - coefficient * self.variables[i+2]
        #self.constraints = [Expression(K_f1)]    
        self.constraints = [Expression(K_f1), Expression(K_f2)]
    
    #########################
    def get_score(self, result, penalty=0):
        # "10000"
        # this function is used to evaluate the quality of the result
        
        x = [int(val) for val in result]
       
        if x[0] + x[1] -1 == 0 and 5 * x[0] + 2 * x[1] <= 5  and 5*x[0] + 2*x[1] - x[2] - 2*x[3] - 2*x[4] == 0:
           # print(x,  2 * x[0] + 5 * x[1]+ x[0] * x[1])
            return 2 * x[0] + 5 * x[1]+ x[0] * x[1]
       # print(x,penalty)
        return penalty
    
    

problem = SimpleProblem()

print(f"Variables used to describe objective function"
      f" and constraints: {problem.variables}")
print(f"Objective function: {problem.objective_function}")
print("Constraints (RHS == 0):")
for constraint in problem.constraints:
    print(f"    {constraint}")
    
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

from QHyper.solvers import Solver
hyper_optimizer_bounds = [(1, 2), (1,2),(1,2)]
solver_config2 = {
    "solver": {
        "type": "vqa",
        "args": {
            "config": {
                "pqc": {
                    "type": "sqaoa",
                    "layers": 5,
                }
            }
        }
        
    }
}
solver_config3= {
    "solver": {
        "type": "vqa",
        "args": {
            "config": {
                "pqc": {
                    "type": "qaoa",
                    "layers": 5,
                },
                'optimizer': {
                    'type': 'scipy',
                    'maxfun': 450,
                },
            }
        }
        
    }
}

vqa2 = Solver.from_config(problem, solver_config2)
vqa3 = Solver.from_config(problem, solver_config3)
params_config = {
        'angles': [[0.4]*5, [0.7]*5], # QAOA angles - first we have gammas (for the cost Hamiltonian), then we have betas (for the mixer)
       # 'hyper_args': [1, 1, 1], #  those are the alpha values from [1]
       'hyper_args': [1, 3.5, 1.5]
    }

if __name__ == '__main__': 
    best_params = vqa2.evaluate(params_config)
    #best_params = vqa3.solve(params_config)
    print(f"Best params: {best_params}")
    
   #best_params=tester.solve(best_params)
    
    best_results = tester.evaluate(best_params,  print_results=True)
    print(f"Best results: {best_results}")
    print(f"Params used for optimizer:\n{best_params['angles']},\n"
           f"and params used for hyperoptimizer: {best_params['hyper_args']}")
