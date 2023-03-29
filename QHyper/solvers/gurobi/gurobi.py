import gurobipy as gp
from gurobipy import GRB

from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.optimizers.base import Optimizer
from QHyper.solvers.converter import QUBO


def calc(vars: dict[str, Any], poly_dict: QUBO) -> Any:
    cost_function: float = 0
    for key, value in poly_dict.items():
        tmp = 1
        for k in key:
            tmp *= vars[k]
        cost_function += tmp * value
    return cost_function


class Gurobi(Solver):  # todo works only for quadratic expressions
    def __init__(self, **kwargs: Any) -> None:
        self.problem: Problem = kwargs["problem"]

    def solve(self, params_inits: dict[str, Any], hyper_optimizer: Optional[Optimizer] = None) -> Any:
        gpm = gp.Model("name")

        vars = {str(var_name): gpm.addVar(vtype=gp.GRB.BINARY, name=str(var_name))
                for var_name in self.problem.variables}
        # gpm.update()

        objective_function = calc(vars, self.problem.objective_function.as_dict())
        gpm.setObjective(objective_function, gp.GRB.MINIMIZE)

        for i, constraint in enumerate(self.problem.constraints):
            tmp_constraint = calc(vars, constraint.as_dict())
            gpm.addConstr(tmp_constraint == 0, f"constr_{i}")
            gpm.update()
            print(tmp_constraint)


        # eq_constraints = self.problem.constraints["=="]
        # for i, constraint in enumerate(eq_constraints):
        #     tmp_constraint = calc(vars, constraint.as_dict())
        #     gpm.addConstr(tmp_constraint == 0, f"eq_constr_{i}")
        #
        # ltq_constraints = self.problem.constraints["<="]
        # for i, constraint in enumerate(ltq_constraints):
        #     tmp_constraint = calc(vars, constraint.as_dict())
        #     gpm.addConstr(tmp_constraint <= 0, f"leq_constr_{i}")
        #
        # gtq_constraints = self.problem.constraints["<="]
        # for i, constraint in enumerate(gtq_constraints):
        #     tmp_constraint = calc(vars, constraint.as_dict())
        #     gpm.addConstr(tmp_constraint >= 0, f"gtq_constr_{i}")

        gpm.optimize()

        # print(dir(gpm))
        # print("status ", gpm.Status)
        # print("is optimal? ", GRB.OPTIMAL)
        #
        # allvars = gpm.getVars()
        # for v in allvars[-1:]:
        #     print(v.VarName)
        #     print(v.X)
        #     print(dir(v))


        allvars = gpm.getVars()
        solution = {}
        for v in allvars:
            solution[v.VarName] = v.X

        return solution
