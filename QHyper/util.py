from _ast import Compare
import ast
import sympy

from typing import Union, cast, Any, Callable
from enum import Enum
from dataclasses import dataclass, field


VARIABLES = Union[tuple[()], tuple[str], tuple[str, str], tuple[str, ...]]
QUBO = dict[VARIABLES, float]

@dataclass
class Component:
    """Represents a mathematical component with separated variables from the numerical coefficient.
    For example, Component(['x1', 'x2'], 2) represents 2 * 'x1' * 'x2'.
    
    Attributes:
    ----------
    variables : A list of variables. 
    coefficient : The coefficient.
    """
    variables: list[str] = field(default_factory=list)
    coefficient: int = field(default=1)
        
    def add_variable(self, variable: str) -> None:
        self.variables.append(variable)
    
    def set_coefficient(self, new_coefficient: int) -> None:
        self.coefficient = new_coefficient



class Parser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.polynomial_as_dict: QUBO = {}

    def visit_Expr(self, node: ast.Expr) -> None:
        visited = self.visit(node.value)

        if isinstance(visited, Component):
            self.polynomial_as_dict[tuple(visited.variables)] = visited.coefficient
        elif isinstance(visited, list):
            for component in visited:
                self.polynomial_as_dict[tuple(component.variables)] = component.coefficient
        else:
            raise Exception("TBD")
            

    def visit_Constant(self, node: ast.Constant) -> Component:
        return Component(coefficient=node.value)
        # return Component([], node.value)

    def visit_Name(self, node: ast.Name) -> Component:
        return Component(variables=[node.id])
        # return Component([node.id], 1)

    def visit_BinOp(self, node: ast.BinOp) -> Component | list[Component]:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
  
        if isinstance(node.op, ast.Add):
            return self.combine_components(lhs, rhs)

        if isinstance(node.op, ast.Sub):
            return self.combine_components(lhs, rhs, is_subtraction=True)

        if isinstance(node.op, ast.Mult):
            return Component(lhs.variables + rhs.variables, lhs.coefficient * rhs.coefficient)

        if isinstance(node.op, ast.Pow):
            return Component([node.left.id for _ in range(node.right.value)], 1) #todo is it really 1?


    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any: #todo what it really returns?
        lhs = self.visit(node.operand)

        if isinstance(node.op, ast.USub):
            previous_coefficient = lhs.coefficient
            lhs.set_coefficient(-1 * previous_coefficient)
            return lhs

        if isinstance(node.op, ast.UAdd): #TODO - check if it is needed
            return lhs


    @staticmethod
    def combine_components(left: Component | list[Component], right: Component | list[Component], 
            is_subtraction: bool = False) -> list[Component]:

        if is_subtraction:
            right.set_coefficient(-1 * right.get_coefficient())

        if isinstance(left, Component) and isinstance(right, Component):
            return [left, right]
        elif isinstance(left, list) and isinstance(right, Component):
            return left + [right]
        elif isinstance(left, list) and isinstance(right, list):
            return left + right
        else:
            raise Exception("TBD")



# class Expression:
    # dict
    # EQ GE LE  

            
class Operator(Enum):
    EQ = "=="
    GE = ">="
    LE = "<="
            

class Constraint:
    def __init__(self, lhs, rhs, operator):
        """ For now, we assume that the constraint is in the form of: number >= sum of something"""
        self.lhs: int | float = lhs
        self.rhs: QUBO = rhs
        self.operator: Operator = operator # for now it is only GE: number >= some polynomial without constants
        
    def setup(self,):
        ...
        

            
class Expression:
    def __init__(self, operator: Operator = Operator.EQ) -> None:
        self.dictionary: dict = {}
        self.operator: Operator = operator
        
        
        def set_operator(new_operator: Operator) -> None:
            self.operator = new_operator
            
        def as_dict(self) -> QUBO:
            if self.polynomial is not None:
                parser = Parser()
                ast_tree = ast.parse(
                    str(sympy.expand(self.polynomial))
                )  # type: ignore[no-untyped-call]
                parser.visit(ast_tree)
                self.dictionary = parser.polynomial_as_dict
            return self.dictionary

    def __repr__(self) -> str:
        return str(self.dictionary)

            
                       
# class Expression:
#     def __init__(self, equation: sympy.core.Expr | sympy.core.Rel | dict) -> None:
#         if isinstance(equation, sympy.core.Expr):
#             self.polynomial: sympy.core.Expr | None = equation
#             self.dictionary: dict = self.as_dict()
#         # elif isinstance(equation, sympy.core.relational.LessThan):
#         #     self.polynomial: sympy.core.Expr | None = sympy.simplify(constraint_eq.lhs - constraint_eq.rhs)
#         #     self.dictionary: dict = self.as_dict()
#         # elif isinstance(equation, sympy.core.relational.GreaterThan):
#         #     self.polynomial: sympy.core.Expr | None = sympy.simplify((-1) * (constraint_eq.lhs - constraint_eq.rhs))
#         #     self.dictionary: dict = self.as_dict()

#         elif isinstance(equation, dict):
#             self.polynomial: sympy.core.Expr | None = None
#             self.dictionary: dict = equation
#         else:
#             raise Exception(
#                 "Expression equation must be an instance of "
#                 "sympy.core.Expr or dict, got of type: "
#                 f"{type(equation)} instead"
#             )

#     def as_dict(self) -> QUBO:
#         if self.polynomial is not None:
#             parser = Parser()
#             ast_tree = ast.parse(
#                 str(sympy.expand(self.polynomial))
#             )  # type: ignore[no-untyped-call]
#             parser.visit(ast_tree)
#             self.dictionary = parser.polynomial_as_dict
#         return self.dictionary

#     def __repr__(self) -> str:
#         return str(self.dictionary)

    # def as_dict_with_slacks(self):
    #     parser = Parser()
    #     objective_function = sympy.expand(self.polynomial)
    #     ast_tree = ast.parse(str(objective_function))
    #     parser.visit(ast_tree)

    #     result = parser.polynomial_as_dict
    #     if self.op == '==':
    #         return result

    #     if self.op == '<=':
    #         if tuple() in result:
    #             value = result[tuple()]
    #             return result | calc_slack_coefficients(value)
    #         return result
    #     else:
    #         raise Exception("Unimplemented")

    # def as_polynomial(self) -> str:
    #     if self.polynomial is not None:
    #         return str(self.polynomial)
    #     else:
    #         polynomial = str()
    #         for k in self.dictionary:
    #             if self.dictionary[k] < 0:
    #                 polynomial += "- "
    #             polynomial += str(abs(self.dictionary[k])) + "*"
    #             polynomial += "*".join(k)
    #             polynomial += " "
    #         return polynomial.rstrip()


def weighted_avg_evaluation(
    results: dict[str, float],
    score_function: Callable[[str, float], float],
    penalty: float = 0,
    limit_results: int | None = None,
    normalize: bool = True,
) -> float:
    score: float = 0

    sorted_results = sort_solver_results(results, limit_results)
    if normalize:
        scaler = 1/sum([v for v in sorted_results.values()])
    else:
        scaler = 1

    for result, prob in sorted_results.items():
        score += scaler * prob * score_function(result, penalty)
    return score


def sort_solver_results(
    results: dict[str, float],
    limit_results: int | None = None,
) -> dict[str, float]:
    limit_results = limit_results or len(results)
    return {
        k: v for k, v in
        sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )[:limit_results]
    }


def add_evaluation_to_results(
    results: dict[str, float],
    score_function: Callable[[str, float], float],
    penalty: float = 1,
) -> dict[str, tuple[float, float]]:
    """
    Parameters
    ----------
    results : dict[str, float]
        dictionary of results
    score_function : Callable[[str, float], float]
        function that receives result and penalty and returns score, probably
        will be passed from Problem.get_score

    Returns
    -------
    dict[str, tuple[float, float]]
        dictionary of results with scores
    """

    return {
        k: (v, score_function(k, penalty))
        for k, v in results.items()
    }

