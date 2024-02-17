# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import cast

import dimod
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from QHyper.structures.polynomial import Polynomial
from QHyper.structures.constraint import (
    Constraint, SLACKS_LOG_2, UNBALANCED_PENALIZATION, Operator)
from QHyper.problems.base import Problem
import numpy as np


class Converter:
    @staticmethod
    def calc_slack_coefficients(constant: int) -> list[int]:
        num_slack = int(np.floor(np.log2(constant)))
        slack_coefficients = [2**j for j in range(num_slack)]
        if constant - 2**num_slack >= 0:
            slack_coefficients.append(constant - 2**num_slack + 1)
        return slack_coefficients

    @staticmethod
    def use_slacks(const: int | float, label: str) -> Polynomial:
        if const <= 0 or not int(const) == const:
            raise ValueError("Const must be a positive integer")
        const = int(const)
        slack_coefficients = Converter.calc_slack_coefficients(const)

        return Polynomial(
            {(f"{label}_{i}",): v
             for i, v in enumerate(slack_coefficients)}
        )

    @staticmethod
    def apply_slacks(
        constraint: Constraint, weight: list[float]
    ) -> Polynomial:
        if len(weight) != 1:
            raise ValueError("Weight must be a list of length 1")

        rhs_without_const, rhs_const = constraint.rhs.separate_const()

        lhs = constraint.lhs - rhs_without_const
        slacks = Converter.use_slacks(rhs_const, constraint.label)

        return weight[0] * (lhs + slacks - rhs_const) ** 2

    @staticmethod
    def use_unbalanced_penalization(
        constraint: Constraint, weight: list[float]
    ) -> Polynomial:
        lhs = constraint.lhs - constraint.rhs
        return weight[0]*lhs + weight[1]*lhs**2

    @staticmethod
    def assign_weights_to_constraints(
        constraints_weights: list[float], constraints: list[Constraint]
    ) -> list[tuple[list[float], Constraint]]:
        weights_constraints_list = []
        idx = 0
        for constraint in constraints:
            if constraint.method_for_inequalities == UNBALANCED_PENALIZATION:
                weights_constraints_list.append(
                    ([constraints_weights[idx], constraints_weights[idx + 1]], constraint)
                )
                idx += 2
            else:
                weights_constraints_list.append(([constraints_weights[idx]], constraint))
                idx += 1
        return weights_constraints_list

    @staticmethod
    def create_qubo(problem: Problem, weights: list[float]) -> Polynomial:
        result = weights[0] * problem.objective_function

        constraints_weights = weights[1:]
        for weight, constraint in Converter.assign_weights_to_constraints(
            constraints_weights, problem.constraints
        ):
            if constraint.operator == Operator.EQ:
                result += weight[0] * (constraint.lhs - constraint.rhs) ** 2
                continue

            lhs = constraint.lhs - constraint.rhs
            if constraint.operator == Operator.GE:
                lhs = -lhs

            if constraint.method_for_inequalities == SLACKS_LOG_2:
                result += Converter.apply_slacks(constraint, weight)
            elif (constraint.method_for_inequalities
                  == UNBALANCED_PENALIZATION):
                result += Converter.use_unbalanced_penalization(
                    constraint, weight
                )

        return result

    @staticmethod
    def to_cqm(problem: Problem) -> ConstrainedQuadraticModel:
        binary_polynomial = dimod.BinaryPolynomial(
            problem.objective_function.dictionary, dimod.BINARY
        )
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        # todo this cqm can probably be initialized in some other way
        for var in problem.variables:
            if str(var) not in cqm.variables:
                cqm.add_variable(dimod.BINARY, str(var))

        for i, constraint in enumerate(problem.constraints):
            cqm.add_constraint(dict_to_list(constraint.lhs),
                               constraint.operator.value, constraint.rhs)

        return cqm

    @staticmethod
    def to_dimod_qubo(problem: Problem, lagrange_multiplier: float = 10
                      ) -> tuple[dict[tuple[str, ...], float], float]:
        cqm = Converter.to_cqm(problem)
        bqm, _ = dimod.cqm_to_bqm(
            cqm, lagrange_multiplier=lagrange_multiplier)
        return cast(tuple[dict[tuple[str, ...], float], float],
                    bqm.to_qubo())  # (qubo, offset)

    @staticmethod
    def to_dqm(problem: Problem) -> DiscreteQuadraticModel:
        dqm = dimod.DiscreteQuadraticModel()

        BIN_OFFSET = 1 if problem.cases == 1 else 0

        def binary_to_discrete(v: str) -> str:
            id = int(v[1:])
            discrete_id = id // problem.cases
            return f"x{discrete_id}"

        variables_discrete = [
            binary_to_discrete(str(v)) for v in problem.variables[:: problem.cases]
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
                    {(case, case): bias for case in range(problem.cases + BIN_OFFSET)},
                )
            else:
                dqm.set_linear(
                    dqm.variables[xi_idx],
                    [bias for _ in range(problem.cases + BIN_OFFSET)],
                )

        return dqm
