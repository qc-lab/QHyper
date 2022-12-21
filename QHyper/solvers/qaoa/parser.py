import ast
import re
from typing import Any

import pennylane as qml
import sympy


class Visitor(ast.NodeVisitor):
    results: list[qml.Hamiltonian]

    def __init__(self) -> None:
        self.results = []

    def visit_Expr(self, node: ast.Expr) -> Any:
        for field, value in ast.iter_fields(node):
            self.results.append(self.visit(value))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            if isinstance(left, qml.Hamiltonian) and isinstance(right, qml.Hamiltonian):
                return left @ right
            else:
                return left * right

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id[0] == "x":
            return self.substitution(int(node.id[1:]))
        if node.id[0] == "J":
            return qml.Identity(-1)

    def visit_Call(self, node: ast.Call) -> Any:
        if node.func.id == "sum":
            result = 0
            for elem in node.args[0].elts:
                result += self.visit(elem)

            return result

    def visit_List(self, node: ast.List) -> Any:
        print(node.elts)

    def substitution(self, wire):
        return 0.5 * qml.Identity(wire) - 0.5 * qml.PauliZ(wire)


def parse_hamiltonian(expresion: str) -> qml.Hamiltonian:
    """Function parsing string to PennyLane Hamiltonian"""

    expresion = str(sympy.expand(expresion))
    # All the variables are binary, so the power can be removed
    expresion = str(sympy.simplify(re.sub(r'\*\*\d*', '', expresion)))
    tree = ast.parse(expresion)
    vis = Visitor()
    vis.visit(tree)
    for i, op in enumerate(vis.results[0].ops):
        if op.wires.tolist() == [-1]:
            return qml.Hamiltonian(
                vis.results[0].coeffs.tolist()[:i] + vis.results[0].coeffs.tolist()[i + 1:],
                vis.results[0].ops[:i] + vis.results[0].ops[i + 1:]
            )
    return vis.results[0]
