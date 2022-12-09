import itertools

from .problem import Problem
import numpy as np
import sympy

class WorkflowtestProblem(Problem):
    """Class defining objective function and constraints for Workflow problem
    
    Attributes
    ----------
    objective_function : str
        objective function in SymPy syntax
    constraints : list[str]
        list of constraints in SymPy syntax
    wires : int
        number of qubits in the circuit, equals to number of cities to the power of 2
    """

    def __init__(
        self, 
    ) -> None:
        """
        Parameters
        ----------
        """        
        self.wires = 13
        self.M = [2, 6, 3]
        self.K = [1, 4, 2]
        self.T = [12, 6, 24]
        self.d = 19
        
        self.time_matrix = np.array(self._get_time_matrix())
        self.cost_matrix = np.array(self._get_cost_matrix())
        self._create_objective_function()
        self._create_constraints()
        print(self.objective_function)
        print(self.constraints)
      
    
    def _create_objective_function(self) -> None:
        x = sympy.symbols(' '.join([f'x{i}' for i in range(self.wires)]))
        self.objective_function = self._get_cost_model(x)
        

    def _create_constraints(self) -> None:
        x = sympy.symbols(' '.join([f'x{i}' for i in range(self.wires)]))
        constraints = []
        constraints.append(sympy.simplify(sympy.expand(self._get_machine_usage_model(x))))
        constraints.append(sympy.simplify(sympy.expand(self._get_deadline_model(x))))
        self.constraints = constraints
        
    def _get_time_matrix(self):
        r = []
        for i in self.M:
            tmp = []
            for j in self.T:
                tmp.append(j / i)
            r.append(tmp)
        return np.array(r)


    def _get_cost_matrix(self):
        m = []
        for i in range(len(self.time_matrix)):
            tmp = []
            for j in self.time_matrix[i]:
                tmp.append(self.K[i] * j)
            m.append(tmp)
        return m

    def _get_cost_model(self,x):
        return sum(
            sum([
                self.cost_matrix[machine_index, task_index] * x[task_index * 3 + machine_index] 
                for machine_index 
                in range(3)
            ])
            for task_index 
            in range(3)
        )


    def _get_machine_usage_model(self,x):     
        return sum(
            (1 - sum([x[task_index * 3 + machine_index] for machine_index in range(3)])) ** 2
            for task_index 
            in range(3)
        )

    def _get_deadline_model(self,x):
        time_sum = sum(
            sum([
                self.time_matrix[machine_index, task_index] * x[task_index * 3 + machine_index] 
                for machine_index 
                in range(3)
            ])
            for task_index 
            in range(3)
        )
        slack_sum = 8 * x[9] + 4 * x[10] + 2 * x[11] + x[12]
        time_constraint = (self.d - time_sum - slack_sum) ** 2
        
        return time_constraint
    
    def get_score(self, result) -> float | None:
        """Returns length of the route based on provided outcome in bits. 
        
        Parameters
        ----------
        result : str
            route as a string of zeros and ones

        Returns
        -------
        float | None
            Returns length of the route, or None if route wasn't correct
        """
        
        return None # Bigger value that possible distance 
      
#wft = WorkflowtestProblem()