from typing import Any, cast

import dimod
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from QHyper.util import Expression
from QHyper.problems.base import Problem
from QHyper.util import QUBO, VARIABLES


def dict_to_list(my_dict: QUBO) -> list[tuple[Any, ...]]:
    return [tuple([*key, val]) for key, val in my_dict.items()]


class Converter:
    @staticmethod
    def create_qubo(problem: Problem, weights: list[float]) -> QUBO:
        results: dict[VARIABLES, float] = {}

        if len(weights) != len(problem.constraints) + 1:
            raise Exception(
                f"Expected {len(problem.constraints)+1} weights, "
                f"got {len(weights)} (weights: {weights}))"
            )

        objective_function = Expression(
            {
                key: weights[0] * val
                for key, val in problem.objective_function.as_dict().items()
            }
        )
        for key, value in objective_function.as_dict().items():
            if key in results:
                results[key] += value
            else:
                results[key] = value

        constraint_weights = weights[1:]

        for weight, element in zip(constraint_weights, problem.constraints):
            constraint = Expression(element.polynomial**2 * weight)
            for key, value in constraint.as_dict().items():
                if key in results:
                    results[key] += value
                else:
                    results[key] = value

        return results

    @staticmethod
    def create_weight_free_qubo(problem: Problem) -> QUBO:
        results: dict[VARIABLES, float] = {}

        objective_function = Expression(problem.objective_function.polynomial)
        for key, value in objective_function.as_dict().items():
            if key in results:
                results[key] += value
            else:
                results[key] = value
        

        return results

    @staticmethod
    def to_cqm(problem: Problem) -> ConstrainedQuadraticModel:
        binary_polynomial = dimod.BinaryPolynomial(
            problem.objective_function.as_dict(), dimod.BINARY
        )
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        # todo this cqm can probably be initialized in some other way
        for var in problem.variables:
            if str(var) not in cqm.variables:
                cqm.add_variable(dimod.BINARY, str(var))

        for i, constraint in enumerate(problem.constraints):
            cqm.add_constraint(
                dict_to_list(constraint.as_dict()), "==", label=i
            )

        return cqm

    @staticmethod
    def to_qubo(problem: Problem) -> tuple[QUBO, float]:
        cqm = Converter.to_cqm(problem)
        bqm, _ = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)
        return cast(tuple[QUBO, float], bqm.to_qubo())  # (qubo, offset)

    @staticmethod
    def to_dqm(problem: Problem) -> DiscreteQuadraticModel:
        dqm = dimod.DiscreteQuadraticModel()

        BIN_OFFSET = 1 if problem.cases == 1 else 0

        def binary_to_discrete(v: str) -> str:
            id = int(v[1:])
            discrete_id = id // problem.cases
            return f"x{discrete_id}"

        variables_discrete = [
            binary_to_discrete(str(v))
            for v in problem.variables[:: problem.cases]
        ]
        for var in variables_discrete:
            if var not in dqm.variables:
                dqm.add_variable(problem.cases + BIN_OFFSET, var)

        for vars, bias in problem.objective_function.as_dict().items():
            s_i, *s_j = vars
            x_i = binary_to_discrete(s_i)
            xi_idx: int = cast(int, dqm.variables.index(x_i))
            if s_j:
                x_j = binary_to_discrete(*s_j)
                xj_idx: int = cast(int, dqm.variables.index(x_j))
                dqm.set_quadratic(
                    dqm.variables[xi_idx],
                    dqm.variables[xj_idx],
                    {
                        (case, case): bias
                        for case in range(problem.cases + BIN_OFFSET)
                    },
                )
            else:
                dqm.set_linear(
                    dqm.variables[xi_idx],
                    [bias for _ in range(problem.cases + BIN_OFFSET)],
                )

        return dqm
