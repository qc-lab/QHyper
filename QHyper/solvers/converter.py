import dimod

from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel

from typing import Any, cast

import sympy

from QHyper.hyperparameter_gen.parser import Expression
from QHyper.problems.base import Problem
from QHyper.problems.community_detection import (
    CommunityDetectionProblem,
)
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
                f"got {len(weights)}"
            )

        objective_function = Expression(
            problem.objective_function.polynomial * weights[0]
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

        if not hasattr(problem, "N_cases"):
            raise Exception(
                "Number of discrete variable values (cases) required for DQM"
            )
        N_cases = problem.N_cases

        def get_discrete_var_name(v: sympy.Symbol | str) -> str:
            return f"x{str(decode_discrete_variable(v))}"

        def decode_discrete_variable(v: sympy.Symbol | str) -> int:
            id = int(str(v)[len("s") :])
            return int(id // problem.N_cases)

        discrete_vars = list(
            set([get_discrete_var_name(v) for v in problem.variables])
        )
        for var in discrete_vars:
            if var not in dqm.variables:
                dqm.add_variable(N_cases, var)

        def dqm_var(var_str_idx: str) -> Any | int:
            return dqm.variables.index(var_str_idx)

        for vars, bias in problem.objective_function.as_dict().items():
            u, *v = vars
            u_idx: int = cast(int, dqm_var(get_discrete_var_name(u)))
            if v:
                v_idx: int = cast(int, dqm_var(get_discrete_var_name(*v)))
                dqm.set_quadratic(
                    dqm.variables[u_idx],
                    dqm.variables[v_idx],
                    {(case, case): bias for case in range(N_cases)},
                )
            else:
                dqm.set_linear(
                    dqm.variables[u_idx], [bias for _ in range(N_cases)]
                )

        return dqm
