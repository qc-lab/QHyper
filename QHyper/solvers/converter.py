import dimod
import sympy

from typing import NewType

from dimod import ConstrainedQuadraticModel

from ..problems.problem import Problem


# QUBO = NewType('QUBO', dict[tuple[int, int], float])
class QUBO(dict):
    variables: list[str]


def dict_to_list(my_dict):
    result = []
    for key, val in my_dict.items():
        tmp = list(key)
        result.append(tuple(tmp + [val]))
    return result


class Converter:
    @staticmethod
    def create_qubo(problem: Problem, weights: list[float]) -> QUBO:
        elements = [problem.objective_function] + problem.constraints
        results = {}
        for weight, element in zip(weights, elements):
            print(sympy.expand(element.polynomial))
            element.polynomial *= weight
            for key, value in element.as_dict().items():
                if key in results:
                    results[key] += value
                else:
                    results[key] = value
        print(results)

        return results

    @staticmethod
    def to_cqm(problem: Problem):
        binary_polynomial = dimod.BinaryPolynomial(problem.objective_function.as_dict(), dimod.BINARY)
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        bla = ConstrainedQuadraticModel()
        for var in problem.variables:  # todo this cqm can probably be initialized in some other way
            if str(var) not in cqm.variables:
                cqm.add_variable(dimod.BINARY, str(var))

        for constraint in problem.constraints:
            constraint = constraint.as_dict()
            constraint = dict_to_list(constraint)
            cqm.add_constraint(constraint, "==")

        return cqm

    @staticmethod
    def to_qubo(problem: Problem) -> tuple[dict[tuple[str, str], float], float]:
        cqm = Converter.to_cqm(problem)
        bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)
        return bqm.to_qubo()  # (qubo, offset)
