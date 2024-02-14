import ast
from typing import Any

from QHyper.structures.polynomial import Polynomial


class ParserException(Exception):
    pass


class Parser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.polynomial: Polynomial = Polynomial()

    def visit_Expr(self, node: ast.Expr) -> None:
        visited = self.visit(node.value)

        return visited

    def visit_Constant(self, node: ast.Constant) -> Polynomial:
        return Polynomial({tuple(): node.value})

    def visit_Name(self, node: ast.Name) -> Polynomial:
        return Polynomial({(node.id,): 1})

    def visit_BinOp(self, node: ast.BinOp) -> Polynomial:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return lhs + rhs

        if isinstance(node.op, ast.Sub):
            return lhs - rhs

        if isinstance(node.op, ast.Mult):
            return lhs * rhs

        if isinstance(node.op, ast.Pow):
            return lhs ** rhs

        raise ParserException(f"Unsupported operation: {node.op}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:  # todo what it really returns?
        lhs = self.visit(node.operand)

        if isinstance(node.op, ast.USub):
            return -lhs

        if isinstance(node.op, ast.UAdd):  # TODO - check if it is needed
            return lhs

