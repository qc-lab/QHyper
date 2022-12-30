import ast
from typing import Any

import sympy


class Expression:  # todo rename to Polynomial?

    def __init__(self, objective_function, constraints):  # here in sympy
        self.objective_function = objective_function
        self.constraints = constraints

    def as_dict(self, expr):
        # objective_function = {}  # here in sympy
        # constraints = {"==": [{}],
        #                "<=": [{}],
        #                ">=": [{}]}
        parser = Parser()
        parsed_objective_function = self.helper(parser, self.objective_function)

        parsed_constraints = {}
        for constraint in self.constraints:
            parser.reset()
            rel_op = constraint.rel_op  # "==" "<=" ">="
            tmp = constraint.lhs - constraint.rhs
            tmp_res = self.helper(parser, tmp)
            parsed_constraints[rel_op].append(tmp_res)

        return parsed_objective_function, parsed_constraints

    @staticmethod
    def helper(parser, expr):
        objective_function = str(sympy.expand(expr))  # after that the constant allways will be last
        ast_tree = ast.parse(objective_function)
        parser.visit(ast_tree)
        return parser.polynomial_as_dict

    def as_string(self):
        pass

    def as_qubo(self):
        pass

    def as_bqm(self):
        pass


class Parser(ast.NodeVisitor):
    def __init__(self):
        self.polynomial_as_dict = {}

    def reset(self):
        self.polynomial_as_dict = {}

    def visit_Expr(self, node: ast.Expr) -> Any:  # todo make sure it is a single expression

        visit_value = self.visit(node.value)

        if isinstance(visit_value, list):
            for i in visit_value:
                self.polynomial_as_dict[tuple(i[0])] = i[1]
        elif isinstance(visit_value, str):  # expression consisting of a single variable name
            self.polynomial_as_dict[(visit_value,)] = 1
        elif isinstance(visit_value, (int, float)):  # expression consisting of a single numerical value
            self.polynomial_as_dict[()] = visit_value
        else:
            raise Exception

    def visit_BinOp(self, node: ast.BinOp) -> Any:

        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return self.tmp(left, right)

        if isinstance(node.op, ast.Sub):
            return self.tmp(left, right, multiplier=-1)

        if isinstance(node.op, ast.Mult):
            if isinstance(left, list):
                variable, constant = left[0]  # unpack it a bit
            else:
                variable, constant = [], 1
                if isinstance(left, str):
                    variable.append(left)
                elif isinstance(left, (int, float)):
                    constant *= left

            if isinstance(right, str):
                variable.append(right)
            elif isinstance(right, (int, float)):
                constant *= right
            elif isinstance(right, list):  # only in the case of powers:
                variable += right[0][0]
            return [[variable, constant]]

        if isinstance(node.op, ast.Pow):
            return [[[node.left.id for _ in range(node.right.value)], 1]]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            if isinstance(operand, (int, float)):
                return [[[], -operand]]
            elif isinstance(operand, list):
                operand[0][1] = -1 * operand[0][1]
                return operand
            else:  # is there any exception case?
                return [[[operand], -1]]

        if isinstance(node.op, ast.UAdd):
            return operand

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        return node.id

    @staticmethod
    def tmp(left, right, multiplier=1):

        if isinstance(left, list):
            if isinstance(right, list):
                right[0][1] = multiplier * right[0][1]  # todo not sure if right should be modifed if it is static
                return left + right
            elif isinstance(right, (int, float)):
                return left + [[[], multiplier * right]]
            elif isinstance(right, str):
                return left + [[[right], multiplier]]
        elif isinstance(right, list):
            right[0][1] = multiplier * right[0][1]
            if isinstance(left, str):
                return [[[left], 1]] + right
            if isinstance(left, (int, float)):
                return [[[], left]] + right
        elif isinstance(right, (int, float)):
            right *= multiplier
            if isinstance(left, str):
                return [[[left], 1]] + [[[], right]]
            if isinstance(left, (int, float)):
                return [[[], left]] + [[[], right]]
        elif isinstance(right, str):
            if isinstance(left, str):
                return [[[left], 1]] + [[[right], multiplier]]
            if isinstance(left, (int, float)):
                return [[[], left]] + [[[right], multiplier]]
        raise Exception


def expression_to_dict(objective_function):
    expression = str(sympy.expand(objective_function))  # after that the constant allways will be last
    tree = ast.parse(expression)
    vis = Parser()
    vis.visit(tree)
    return vis.polynomial_as_dict


def check_if_valid_expression(tree):
    if len(tree.body) != 1:
        print("too many statements, please use a single statement")
    else:
        print("ok")

    if not isinstance(tree.body[0], ast.Expr):
        print("Not a valid expression")
    else:
        print("ok")


def check_if_valid_constraint(tree):
    if (len(tree.body[0].value.ops) != 1 or
            not isinstance(tree.body[0].value.comparators[0], ast.Constant) or
            tree.body[0].value.comparators[0].value != 0):
        print("Not a valid constraint")
    else:
        print("ok")


def tests():  # todo will be moved to a test section of the project
    assert expression_to_dict("8") == {(): 8}
    assert expression_to_dict("-8") == {(): -8}
    assert expression_to_dict("8.8") == {(): 8.8}
    assert expression_to_dict("-8.8") == {(): -8.8}
    assert expression_to_dict("x1") == {('x1',): 1}
    assert expression_to_dict("-x1") == {('x1',): -1}

    assert expression_to_dict("x1 + x2") == {('x1',): 1, ('x2',): 1}
    assert expression_to_dict("1 + x1") == {('x1',): 1, (): 1}
    assert expression_to_dict("1.0 + x1") == {('x1',): 1, (): 1.0}
    assert expression_to_dict("x1 + x2 + 1") == {('x1',): 1, ('x2',): 1, (): 1}
    assert expression_to_dict("x1 + x2 + x3") == {('x1',): 1, ('x2',): 1, ('x3',): 1}

    assert expression_to_dict("x1 - x2") == {('x1',): 1, ('x2',): -1}
    assert expression_to_dict("-x1 + x2") == {('x1',): -1, ('x2',): 1}
    assert expression_to_dict("-x1 - x2") == {('x1',): -1, ('x2',): -1}
    assert expression_to_dict("-x1 - x2**2") == {('x1',): -1, ('x2', 'x2'): -1}

    assert expression_to_dict("x1 * x2") == {('x1', 'x2'): 1}
    assert expression_to_dict("x1 * x2 * 2") == {('x1', 'x2'): 2}
    assert expression_to_dict("x1 * x2 * 2 * x3") == {('x1', 'x2', 'x3'): 2}
    assert expression_to_dict("x1 * x2 + 1") == {('x1', 'x2'): 1, (): 1}
    assert expression_to_dict("1 + x1 * x2") == {('x1', 'x2'): 1, (): 1}
    assert expression_to_dict("x1 + x2 * x3") == {('x2', 'x3'): 1, ('x1',): 1}
    assert expression_to_dict("x1 * x2 + x3") == {('x1', 'x2'): 1, ('x3',): 1}
    assert expression_to_dict("x1 * x2 + x3 * x4") == {('x1', 'x2'): 1, ('x3', 'x4'): 1}

    assert expression_to_dict("x1 * x2 * 2.5") == {('x1', 'x2'): 2.5}
    assert expression_to_dict("x1 * x2 * 1.3 * x3") == {('x1', 'x2', 'x3'): 1.3}

    assert expression_to_dict("x1**2") == {('x1', 'x1'): 1}
    assert expression_to_dict("-x1**2") == {('x1', 'x1'): -1}
    assert expression_to_dict("(1 - (x1 + x2 + x3))**2") == {('x1', 'x1'): 1, ('x1', 'x2'): 2, ('x1', 'x3'): 2,
                                                             ('x1',): -2, ('x2', 'x2'): 1, ('x2', 'x3'): 2, ('x2',): -2,
                                                             ('x3', 'x3'): 1, ('x3',): -2, (): 1}
