# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Any, cast

import dimod
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from QHyper.util import Expression
from QHyper.problems.base import Problem
from QHyper.util import QUBO, VARIABLES, Constraint, MethodsForInequalities, Operator
import numpy as np
import copy


def dict_to_list(my_dict: QUBO) -> list[tuple[Any, ...]]:
    return [tuple([*key, val]) for key, val in my_dict.items()]


def multiply_dicts_sorted(dict_1: dict[str, float], dict_2: dict[str, float]) -> QUBO:
    result = {}
    for key_1, value_1 in dict_1.items():
        for key_2, value_2 in dict_2.items():
            new_key = tuple(sorted(key_1 + key_2))
            if new_key in result:
                result[new_key] += value_1 * value_2
            else:
                result[new_key] = value_1 * value_2

    return result


def multiply_dict_by_constant(dict: QUBO, constant: int | float) -> QUBO:
    return {key: value * constant for key, value in dict.items()}


class Converter:
    @staticmethod
    def calc_slack_coefficients(constant: int | float) -> list[int]:
        num_slack = int(np.floor(np.log2(constant)))
        slack_coefficients = [2**j for j in range(num_slack)]
        if constant - 2**num_slack >= 0:
            slack_coefficients.append(constant - 2**num_slack + 1)
        return slack_coefficients

    @staticmethod
    def get_variables(qubo: QUBO) -> VARIABLES:
        tmp_list = []

        for key in qubo.keys():
            for var in key:
                if var not in tmp_list and var != tuple():
                    tmp_list.append(var)
        return tmp_list

    @staticmethod
    def use_slacks(constraint: Constraint) -> QUBO:
        # todo constraint.lhs cannot have any numerical values
        if constraint.rhs <= 0 or not int(constraint.rhs) == constraint.rhs:
            raise ValueError("Constraint rhs must be a positive integer")

        slack_coefficients = Converter.calc_slack_coefficients(constraint.rhs)
        slack_names = tuple(
            (f"{constraint.label}_{i}",) for i in range(len(slack_coefficients))
        )

        slacks_as_dict = dict(zip(slack_names, slack_coefficients))
        return slacks_as_dict

    @staticmethod
    def apply_slacks(
        results: QUBO, constraint: Constraint, weight: list[float]
    ) -> QUBO:
        if constraint.rhs <= 0 or not int(constraint.rhs) == constraint.rhs:
            raise ValueError("Constraint rhs must be a positive integer")

        constraint_tmp = copy.deepcopy(constraint.lhs)
        constraint_tmp[tuple()] = -constraint.rhs
        slacks = Converter.use_slacks(constraint)
        qubo_with_slakcs = Converter.add_dicts(constraint_tmp, slacks)

        qubo_with_slacks_squared = multiply_dicts_sorted(
            qubo_with_slakcs, qubo_with_slakcs
        )

        weighted_qubo_with_slacks_squared = multiply_dict_by_constant(
            qubo_with_slacks_squared, weight
        )

        return Converter.add_dicts(results, weighted_qubo_with_slacks_squared)

    @staticmethod
    def use_unbalanced_penalization(
        results: QUBO, constraint: Constraint, weight: list[float]
    ) -> QUBO:
        constraints_unbalanced = copy.deepcopy(constraint.lhs)

        if tuple() in constraints_unbalanced:
            constraints_unbalanced[tuple()] -= constraint.rhs
        else:
            constraints_unbalanced[tuple()] = -constraint.rhs

        linear = multiply_dict_by_constant(constraints_unbalanced, weight[0])

        results = Converter.add_dicts(results, linear)

        quadratic = multiply_dicts_sorted(
            constraints_unbalanced, constraints_unbalanced
        )

        quadratic_with_weight = multiply_dict_by_constant(quadratic, weight[1])

        final_results = Converter.add_dicts(results, quadratic_with_weight)
        return final_results

    @staticmethod
    def add_dicts(dict_1: QUBO, dict_2: QUBO) -> QUBO:
        result = copy.deepcopy(dict_1)

        for key, value in dict_2.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
        return result

    @staticmethod
    def assign_weights_to_constraints(
        constraints_weights: list[float], constraints: list[Constraint]
    ):
        # todo add error handling
        tmp_list = []

        idx = 0
        for constraint in constraints:
            if (
                constraint.method_for_inequalities
                == MethodsForInequalities.UNBALANCED_PENALIZATION
            ):
                tmp_list.append(
                    ([constraints_weights[idx], constraints_weights[idx + 1]], constraint)
                )
                idx += 2
            else:
                tmp_list.append((constraints_weights[idx], constraint))
                idx += 1
        return tmp_list

    @staticmethod
    def create_qubo(problem: Problem, weights: list[float]) -> QUBO:
        # todo if we are to append slacks here, to create the circuit size we can't use problem.variables,
        # because now, there will be more
        results: QUBO = {}

        # 1. Process objective function
        objective_function_weight = weights[0]
        for key, value in problem.objective_function.dictionary.items():
            if key in results:
                results[key] += objective_function_weight * value
            else:
                results[key] = objective_function_weight * value

        # 2. Process constraints
        constraints_weights = weights[1:]
        for weight, constraint in Converter.assign_weights_to_constraints(
            constraints_weights, problem.constraints
        ):
            if constraint.operator == Operator.EQ:
                constraint_tmp = copy.deepcopy(constraint.lhs)
                if tuple() in constraint_tmp:
                    # todo what to do with 0
                    constraint_tmp[tuple()] -= constraint.rhs
                else:
                    constraint_tmp[tuple()] = -constraint.rhs

                quadratic = multiply_dicts_sorted(constraint_tmp, constraint_tmp)
                quadratic_with_weight = multiply_dict_by_constant(quadratic, weight)
                results = Converter.add_dicts(results, quadratic_with_weight)

            elif constraint.operator == Operator.LE:
                if (
                    constraint.method_for_inequalities
                    == MethodsForInequalities.SLACKS_LOG_2
                ):
                    results = Converter.apply_slacks(results, constraint, weight)

                elif (
                    constraint.method_for_inequalities
                    == MethodsForInequalities.UNBALANCED_PENALIZATION
                ):
                    results = Converter.use_unbalanced_penalization(
                        results, constraint, weight
                    )

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
