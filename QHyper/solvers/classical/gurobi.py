# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Any, Optional

import gurobipy as gp
from QHyper.problems.base import Problem
from QHyper.solvers.base import Solver
from QHyper.polynomial import Polynomial
from QHyper.constraint import Operator


def polynomial_to_gurobi(gurobi_vars: dict[str, Any], poly: Polynomial) -> Any:
    cost_function_1: float = 0
    for vars, coeff in poly.terms.items():
        tmp = 1
        for v in vars:
            tmp *= gurobi_vars[v]
        cost_function_1 += tmp * coeff
    return cost_function_1


class Gurobi(Solver):  # todo works only for quadratic expressions
    """
    Gurobi solver class.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    model_name : str, optional
        The name of the gurobi model.
    mip_gap : float | None, optional
        The MIP gap.
    suppress_output : bool, optional, default=True
        If True, the solver's output will be suppressed.
    threads : int, optional, default=0
        The number of threads to be used by the solver.
    """

    def __init__(self, problem: Problem, model_name: str = "",
                 mip_gap: float | None = None, suppress_output: bool = True,
                 threads: int = 0) -> None:
        self.problem = problem
        self.model_name = model_name
        self.mip_gap = mip_gap
        self.supress_output = suppress_output
        self.threads = threads

    def solve(self, params_inits: Optional[dict[str, Any]] = None) -> Any:
        if self.supress_output:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
        else:
            env = None

        gpm = gp.Model(self.model_name, env=env)

        if self.mip_gap:
            gpm.Params.MIPGap = self.mip_gap

        gpm.setParam('Threads', self.threads)

        vars = {
            str(var_name): gpm.addVar(vtype=gp.GRB.BINARY, name=str(var_name))
            for var_name in self.problem.objective_function.get_variables().union(
                [con.get_variables() for con in self.problem.constraints]
            )
        }

        objective_function = polynomial_to_gurobi(
            vars, self.problem.objective_function
        )
        gpm.setObjective(objective_function, gp.GRB.MINIMIZE)

        for i, constraint in enumerate(self.problem.constraints):
            lhs = polynomial_to_gurobi(vars, constraint.lhs)
            rhs = polynomial_to_gurobi(vars, constraint.rhs)
            if constraint.operator == Operator.EQ:
                gpm.addConstr(lhs == rhs, f"constr_{i}")
            elif constraint.operator == Operator.LE:
                gpm.addConstr(lhs <= rhs, f"constr_{i}")
            elif constraint.operator == Operator.GE:
                gpm.addConstr(lhs >= rhs, f"constr_{i}")

            gpm.update()
        gpm.optimize()

        allvars = gpm.getVars()
        solution = {}
        for v in allvars:
            solution[v.VarName] = v.X

        return solution
