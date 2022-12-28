import dimod

from QHyper.hyperparameter_gen.parser import Expression, expression_to_dict
from QHyper.problems.workflow_scheduling import Workflow, WorkflowSchedulingProblem


class DimodModel:

    def __init__(self, expression):
        self.expression = expression

    def dict_to_qubo(self):
        binary_polynomial = dimod.BinaryPolynomial(self.expression.objective_function, dimod.BINARY)
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        for sense, constraints in self.expression.constraints.items():
            for constraint in constraints:
                constraint = self.dict_to_list(constraint) #todo check what is wrong with adding dicts
                cqm.add_constraint(constraint, sense)  # dispatches to: add_constraint_from_iterable

        bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)

        qubo, offset = bqm.to_qubo()
        poly = bqm.to_polystring()

        print("binary_polynomial ", binary_polynomial)
        return qubo, poly

    @staticmethod
    def dict_to_list(my_dict):
        result = []
        for key, val in my_dict.items():
            tmp = list(key)
            result.append(tuple(tmp + [val]))
        return result


if __name__ == "__main__":
    workflow = Workflow()
    wsp = WorkflowSchedulingProblem(workflow)

    expression = Expression()
    expression.objective_function = expression_to_dict(str(wsp.objective_function))

    for constraint in wsp.constraints:
        rel_op = constraint.rel_op  # "==" "<=" ">="
        tmp = constraint.lhs - constraint.rhs

        tmp_res = expression_to_dict(tmp)
        expression.constraints[rel_op].append(tmp_res)

    print("objective_function")
    print(expression.objective_function)
    print("constraints")
    print(expression.constraints)

    dimod_model = DimodModel(expression)
    qubo, poly = dimod_model.dict_to_qubo()
    print(qubo)
    print(poly, type(poly))

    wsp.objective_function = poly
    wsp.constraints = []

    # solver = QAOA(
    #     problem=wft,
    #     platform="pennylane",
    #     optimizer=QmlGradientDescent(200, qml.AdamOptimizer(stepsize=0.05)),
    #     layers=5,
    #     weights=[1,],
    #     angles=[[0.5] * 5, [0.5] * 5],
    #     # mixer: str=,
    #     # backend=
    # )

    # value, params, weights = solver.solve()
