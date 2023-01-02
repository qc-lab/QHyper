import gurobipy as gp
import sympy

from QHyper.problems.problem import Problem
from QHyper.problems.workflow_scheduling import Workflow, WorkflowSchedulingProblem
from QHyper.solvers.solver import Solver


class Gurobi(Solver):  # todo works only for quadratic expressions
    def __init__(self, **kwargs) -> None:
        self.problem: Problem = kwargs.get("problem")

    def solve(self):
        gpm = gp.Model("name")

        # print("variables ", self.problem.variables, type(self.problem.variables), type(self.problem.variables[0]))
        vars = {var_name: gpm.addVar(vtype=gp.GRB.BINARY, name=str(var_name)) for var_name in self.problem.variables}
        gpm.update()
        print(vars)
        print(type(vars.keys()))

        print((self.problem.objective_function.polynomial))
        obj_fun = self.problem.objective_function.as_dict()
        print("obj_fun ", obj_fun)

        cost_function = 0
        for key, value in obj_fun.items():  # todo
            print(key, value)
            tmp = 1
            for k in key:
                tmp *= vars[k]
            cost_function += tmp * value
        #     print(key, value)

        # gpm.update()
        # print(gpm)
        gpm.optimize()
