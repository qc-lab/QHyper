import ast
from typing import Any

import sympy


class Expression:

    def __init__(self):
        self.objective_function = {}
        self.constraints = {"==": [],  # todo change for a list only if needed
                            "<=": [],
                            ">=": []}

    def to_dict(self):
        pass

    def to_string(self):
        pass


class Parser(ast.NodeVisitor):
    def __init__(self):
        self.polynomial_as_dict = {}

    def visit_Expr(self, node: ast.Expr) -> Any:

        for field, value in ast.iter_fields(node):  # todo make sure it is a single expression
            visit_value = self.visit(value)

            if isinstance(visit_value, str):  # case of a single variable name
                # print('visit_value', visit_value, type(visit_value))
                self.polynomial_as_dict[(visit_value,)] = 1
                # print('dict', self.polynomial_as_dict)
            elif isinstance(visit_value, int):
                self.polynomial_as_dict[()] = visit_value
            else:
                for i in visit_value:
                    if isinstance(i[0], list):
                        if not i[1]:
                            i[1] = [1]
                        self.polynomial_as_dict[tuple(i[0])] = i[1][0]
                    elif isinstance(i[0], str):
                        self.polynomial_as_dict[((i[0]),)] = 1
                    elif isinstance(i[0], (int, float)):
                        self.polynomial_as_dict[tuple()] = i[0]

    def visit_BinOp(self, node: ast.BinOp) -> Any:

        if isinstance(node.op, ast.Add):
            left = self.visit(node.left)
            right = self.visit(node.right)

            if isinstance(left, list):
                if isinstance(right, list):
                    result = left + right
                elif isinstance(right, (str, int, float)):
                    result = left + [[right]]
                else:
                    raise Exception
            else:
                if isinstance(right, list):
                    result = [[left]] + right
                elif isinstance(right, (str, int, float)):
                    result = [[left], [right]]
                else:
                    raise Exception
            return result

        if isinstance(node.op, ast.Sub):
            left = self.visit(node.left)
            right = self.visit(node.right)

            if isinstance(left, list):
                if isinstance(right, list):
                    right[0][1][0] = (-1) * right[0][1][0]  # change the sign
                    result = left + right
                elif isinstance(right, (int, float)):
                    result = left + [[-right]]
                elif isinstance(right, str):
                    result = left + [[[right], [-1]]]
                else:
                    raise Exception
            else:  # either str or int/float #right is for sure "an atom"
                if isinstance(right, list):
                    right[0][1][0] = (-1) * right[0][1][0]
                    result = [[left]] + right
                elif isinstance(right, (int, float)):
                    result = [[left], [-1 * right]]
                elif isinstance(right, str):
                    result = [[left], [[right], [-1]]]
                else:
                    raise Exception
            return result

        if isinstance(node.op, ast.Mult):
            left = self.visit(node.left)
            right = self.visit(node.right)

            if isinstance(left, list):
                variable, constant = left[0]
            else:
                variable, constant = [], []
                if isinstance(left, str):
                    variable.append(left)
                    constant.append(1)
                elif isinstance(left, (int, float)):
                    constant.append(left)

            if isinstance(right, str):
                variable.append(right)
            elif isinstance(right, (int, float)):
                constant.append(right)
            elif isinstance(right, list):  # only in the case of powers:
                variable += right[0][0]
            return [[variable, constant]]

        if isinstance(node.op, ast.Pow):
            return [[[node.left.id for _ in range(node.right.value)], [1]]]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            if isinstance(operand, (int, float)):
                return -operand
            elif isinstance(operand, list):
                operand[0][1][0] = (-1) * operand[0][1][0]
                return operand
            else:
                return [[[operand], [-1]]]
        if isinstance(node.op, ast.UAdd):
            return operand

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        return node.id


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
    assert expression_to_dict("x1") == {('x1',): 1}
    assert expression_to_dict("-x1") == {('x1',): -1}

    assert expression_to_dict("x1 + x2") == {('x1',): 1, ('x2',): 1}
    assert expression_to_dict("1 + x1") == {('x1',): 1, (): 1}
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

    assert expression_to_dict("x1**2") == {('x1', 'x1'): 1}
    assert expression_to_dict("-x1**2") == {('x1', 'x1'): -1}
    assert expression_to_dict("(1 - (x1 + x2 + x3))**2") == {('x1', 'x1'): 1, ('x1', 'x2'): 2, ('x1', 'x3'): 2,
                                                             ('x1',): -2, ('x2', 'x2'): 1, ('x2', 'x3'): 2, ('x2',): -2,
                                                             ('x3', 'x3'): 1, ('x3',): -2, (): 1}
