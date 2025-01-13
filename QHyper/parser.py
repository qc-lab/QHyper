"""Module for parsing sympy expressions to :py:class:`~QHyper.polynomial.Polynomial`.

This module provides a way to parse sympy expressions and string to 
:py:class:`~QHyper.polynomial.Polynomial`.
It is not recommended to use this module for large polynomials as it is very
slow in comparison to creating Polynomial directly from the dict.

.. rubric:: Functions

.. autofunction:: from_str
.. autofunction:: from_sympy
.. autofunction:: to_sympy

"""

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
    """Method to convert a polynomial to a sympy expression.
    Might be handy to display the polynomial in a more readable form.

    Parameters
    ----------
    poly : Polynomial
        The polynomial to be converted.
    
    Returns
    -------
    str
        The sympy expression.
    """

    polynomial = ""
    for term, const in poly.terms.items():
        if const < 0:
            polynomial += f"{const}*"
        else:
            polynomial += f"+{const}*"
        polynomial += "*".join(term)
    return sympy.parse_expr(polynomial, evaluate=False)


def from_str(equation: str) -> Polynomial:
    """Method to parse a string to a polynomial.
    Uses ast parser to parse the equation. This method is very slow in 
    comparison to creating Polynomial directly from the dict. Although, for 
    smaller polynomials, it is not noticeable.

    Parameters
    ----------
    equation : str
        The equation to be parsed in form of string.

    Returns
    -------
    Polynomial
        The parsed polynomial.
    """

    parser = Parser()
    ast_tree = ast.parse(equation)
    parser.visit(ast_tree)
    if parser.polynomial is None:
        raise ParserException(f"Failed to parse: {equation}")

    return parser.polynomial


def from_sympy(equation: sympy.core.Expr) -> Polynomial:
    """Method to convert a sympy expression to a polynomial.
    Uses ast parser to parse the equation. This method is very slow in 
    comparison to creating Polynomial directly from the dict. Although, for 
    smaller polynomials, it is not noticeable.

    Parameters
    ----------
    equation : sympy.core.Expr
        The sympy expression to be converted.
    
    Returns
    -------
    Polynomial
        The converted polynomial.
    """

    return from_str(str(sympy.expand(equation)))
