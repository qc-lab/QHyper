import re
import ast
import sympy
import pennylane as qml

from typing import Any


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
            # if isinstance(right, qml.Hamiltonian) and left == 1:
            #     left = qml.Identity(right.wires)
            return left + right
        if isinstance(node.op, ast.Sub):
            # if isinstance(right, qml.Hamiltonian) and left == 1:
            #     left = qml.Identity(right.wires)
            return left - right
        if isinstance(node.op, ast.Mult):
            if isinstance(left, qml.Hamiltonian) and isinstance(right, qml.Hamiltonian):
                return left @ right
            else:
                return left * right
        # if isinstance(node.op, ast.Pow):
        #     return self.multiply_hamiltonians(left, left)
    
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
            return qml.Identity("a")
    
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
    expresion = str(sympy.expand(expresion))
    expresion = str(sympy.simplify(re.sub(r'\*\*\d*', '', expresion)))
    tree = ast.parse(expresion)
    vis = Visitor()
    vis.visit(tree)
    return vis.results[0]
