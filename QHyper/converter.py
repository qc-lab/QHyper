# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Any, cast

import dimod
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from QHyper.polynomial import Polynomial
from QHyper.constraint import (
    Constraint, SLACKS_LOG_2, UNBALANCED_PENALIZATION, Operator)
from QHyper.problems.base import Problem
import numpy as np

def dict_to_list(my_dict) -> list[tuple[Any, ...]]:
    return [tuple([*key, val]) for key, val in my_dict.items()]


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
    def create_qubo(problem: Problem, weights: list[float]) -> Polynomial:
        of_weight = weights[0] if len(weights) else 1
        result = float(of_weight) * problem.objective_function

        constraints_weights = weights[1:]
        for weight, constraint in Converter.assign_weights_to_constraints(
            constraints_weights, problem.constraints
        ):
            if constraint.operator == Operator.EQ:
                result += float(weight[0]) * (constraint.lhs - constraint.rhs) ** 2
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
    def create_weight_free_qubo(problem: Problem):
        results = {}
        
        for key, value in problem.objective_function.items():
            if key in results:
                results[key] += value
            else:
                results[key] = value

        return results

    # TODO: Refactor is needed
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

    # TODO: Refactor is needed
    @staticmethod
    def to_dqm(problem: Problem) -> DiscreteQuadraticModel:
        if hasattr(problem, "one_hot_encoding"):
            if problem.one_hot_encoding:
                return Converter._dqm_from_one_hot_representation(problem)
            return Converter._dqm_from_discrete_representation(problem)
        return Converter._dqm_from_one_hot_representation(problem)

    @staticmethod
    def _dqm_from_discrete_representation(problem: Problem) -> DiscreteQuadraticModel:
        dqm = dimod.DiscreteQuadraticModel()

        for var in problem.variables:
            var = str(var)
            if var not in dqm.variables:
                dqm.add_variable(problem.cases, var)

        for vars, bias in problem.objective_function.as_dict().items():
            x_i, x_j = vars
            xi_idx: int = cast(int, dqm.variables.index(x_i))
            xj_idx: int = cast(int, dqm.variables.index(x_j))

            # We're skipping the linear terms
            if xi_idx == xj_idx:
                continue

            dqm.set_quadratic(
                dqm.variables[xi_idx],
                dqm.variables[xj_idx],
                {(case, case): bias for case in range(problem.cases)},
            )

        return dqm

    @staticmethod
    def _dqm_from_one_hot_representation(problem: Problem) -> DiscreteQuadraticModel:
        dqm = dimod.DiscreteQuadraticModel()

        CASES_OFFSET = 1 if problem.cases == 1 else 0

        def binary_to_discrete(v: str) -> str:
            id = int(v[1:])
            discrete_id = id // problem.cases
            return f"x{discrete_id}"

        variables_discrete = [
            binary_to_discrete(str(v)) for v in problem.variables[:: problem.cases]
        ]
        for var in variables_discrete:
            if var not in dqm.variables:
                dqm.add_variable(problem.cases + CASES_OFFSET, var)

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
                    {(case, case): bias for case in range(problem.cases + CASES_OFFSET)},
                )
            else:
                dqm.set_linear(
                    dqm.variables[xi_idx],
                    [bias for _ in range(problem.cases + CASES_OFFSET)],
                )

        return dqm
