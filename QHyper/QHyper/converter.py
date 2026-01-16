# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


"""This module contains the Converter class with static methods for converting Problem objects to different formats.

To use different solvers, but with the same Problem representation, some form 
of conversion is needed. This module provides methods for converting Problem
objects to ConstrainedQuadraticModel, DiscreteQuadraticModel, and QUBO formats.

.. autosummary:: 
    :toctree: generated
    
    Converter

"""


from typing import cast

import dimod
import re
import warnings
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from QHyper.polynomial import Polynomial
from QHyper.constraint import (
    Constraint, SLACKS_LOG_2, UNBALANCED_PENALIZATION, Operator)
from QHyper.problems.base import Problem
import numpy as np


class ProblemWarning(Warning):
    pass


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
    def assign_penalty_weights_to_constraints(
        constraints_weights: list[float], constraints: list[Constraint]
    ) -> list[tuple[list[float], Constraint]]:
        weights_constraints_list = []
        idx = 0
        group_to_weight: dict[int, list[float]] = {}
        for constraint in constraints:
            if constraint.method_for_inequalities == UNBALANCED_PENALIZATION:
                if constraint.group == -1:
                    weights = constraints_weights[idx: idx + 2]
                    idx += 2
                elif constraint.group in group_to_weight:
                    weights = group_to_weight[constraint.group]
                else:
                    weights = constraints_weights[idx: idx + 2]
                    group_to_weight[constraint.group] = weights
                    idx += 2

                weights_constraints_list.append(
                    (weights, constraint)
                )
            else:
                if constraint.group == -1:
                    weights = [constraints_weights[idx]]
                    idx += 1
                elif constraint.group in group_to_weight:
                    weights = group_to_weight[constraint.group]
                else:
                    weights = [constraints_weights[idx]]
                    group_to_weight[constraint.group] = weights
                    idx += 1
                weights_constraints_list.append((weights, constraint))
        return weights_constraints_list

    @staticmethod
    def create_qubo(problem: Problem, penalty_weights: list[float]) -> Polynomial:
        of_weight = penalty_weights[0] if len(penalty_weights) else 1
        result = float(of_weight) * problem.objective_function

        constraints_penalty_weights = penalty_weights[1:]
        for weight, constraint in Converter.assign_penalty_weights_to_constraints(
            constraints_penalty_weights, problem.constraints
        ):
            if constraint.operator == Operator.EQ:
                result += float(weight[0]) * (
                    constraint.lhs - constraint.rhs) ** 2
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
            problem.objective_function.terms, dimod.BINARY
        )
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        variables = problem.objective_function.get_variables()
        for constraint in problem.constraints:
            variables.update(constraint.lhs.get_variables())

        for variable in variables:
            cqm.add_variable(dimod.BINARY, str(variable))

        for i, constraint in enumerate(problem.constraints):
            lhs = [tuple([*key, value])
                   for key, value in constraint.lhs.terms.items()]
            cqm.add_constraint(lhs, constraint.operator.value, label=i)

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
    def to_dqm(problem: Problem, cases: int = 1) -> DiscreteQuadraticModel:
        """
        Convert problem to DQM format.

        Attributes
        ----------
        problem : Problem
            The problem to be solved. Objective funtion variables
            should be written in the format <str><int> (e.g. x10, yz1).
        cases: int, default 1
            Number of variable cases (values)
            1 is denoting binary variable.
        """

        def binary_to_discrete(v: str) -> str:
            for i in range(len(v)):
                if v[i].isdigit():
                    break

            prefix = v[:i]
            numeric_part = v[i:]

            discrete_id = int(numeric_part) // cases
            new_v = prefix + str(discrete_id)

            return new_v

        def extract_number(element) -> int:
            match = re.search(r'(\d+)', element)
            prefix = element[:match.start()]
            number = int(match.group(1))

            return (prefix, number)

        if problem.constraints:
            warnings.warn(
                "Defined problem has constraints. DQM does not support"
                " constraints, it only supports objective functions!",
                ProblemWarning
            )

        pattern = re.compile(r'^[a-zA-Z]+\d+$')
        for variable in problem.objective_function.get_variables():
            if not pattern.match(variable):
                raise ValueError(
                    f"Objective funtion variable '{variable}'"
                    "should be written in the format <str><int> (e.g. x10, yz1)."
                )

        dqm = dimod.DiscreteQuadraticModel()
        objective_function_variables = sorted(
            problem.objective_function.get_variables(), key=extract_number)

        variables = [
            binary_to_discrete(str(v))
            for v in objective_function_variables[:: cases]
        ]
        cases_offset = cases == 1

        for variable in variables:
            if variable not in dqm.variables:
                dqm.add_variable(cases + cases_offset, variable)

        for vars, bias in problem.objective_function.terms.items():
            s_i, *s_j = vars
            x_i = binary_to_discrete(s_i)
            xi_idx: int = cast(int, dqm.variables.index(x_i))
            if s_j:
                x_j = binary_to_discrete(*s_j)
                xj_idx: int = cast(int, dqm.variables.index(x_j))
                dqm.set_quadratic(
                    dqm.variables[xi_idx],
                    dqm.variables[xj_idx],
                    {(case, case): bias for case in range(cases + cases_offset)},
                )
            else:
                dqm.set_linear(
                    dqm.variables[xi_idx],
                    [bias] * (cases + cases_offset),
                )

        return dqm
