import ast
import sympy

from QHyper.polynomial import Polynomial


class ParserException(Exception):
    pass


class Parser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.polynomial: Polynomial | None = None

    def visit_Expr(self, node: ast.Expr) -> None:
        self.polynomial = self.visit(node.value)

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

        raise ParserException(f"Unsupported operation: {lhs} {node.op} {rhs}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Polynomial:
        lhs = self.visit(node.operand)

        if isinstance(node.op, ast.USub):
            return -lhs

        if isinstance(node.op, ast.UAdd):
            return lhs

        raise ParserException(f"Unsupported operation: {node.op}{lhs}")


def to_sympy(poly: Polynomial) -> str:
    polynomial = ""
    for term, const in poly.terms.items():
        if const < 0:
            polynomial += f"{const}*"
        else:
            polynomial += f"+{const}*"
        polynomial += "*".join(term)
    return sympy.parse_expr(polynomial, evaluate=False)


def from_str(equation: str) -> Polynomial:
    parser = Parser()
    ast_tree = ast.parse(equation)
    parser.visit(ast_tree)
    if parser.polynomial is None:
        raise ParserException(f"Failed to parse: {equation}")

    return parser.polynomial


def from_sympy(equation: sympy.core.Expr) -> Polynomial:
    return from_str(str(sympy.expand(equation)))
