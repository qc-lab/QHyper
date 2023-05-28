import sympy
from QHyper.problems.base import Problem
import networkx as nx
from QHyper.hyperparameter_gen.parser import Expression
from sympy.core.expr import Expr
from typing import cast


class MaxCut:
    def __init__(self):
        self.G = nx.Graph()

    def add_edge(self, u, v):
        self.G.add_edge(u, v)

    def create_sample_graph(self):
        self.G = nx.Graph()
        self.G.add_edges_from([(1,2),(1,3),(2,4),(3,4),(3,5),(4,5)])


class MaxCutProblem(Problem):
    def __init__(self, max_cut: MaxCut, cuts_num: int = 2):
        self.max_cut_graph = max_cut.G
        self.num_cases = cuts_num
        self._set_variables()
        self._set_objective_function()
        self.constraints = []

    
    def _set_variables(self) -> None:
        self.variables = sympy.symbols(' '.join(
            [f'x{i}' for i in range(len(self.max_cut_graph.nodes))]
        ))

    def _set_objective_function(self) -> None:
        """
        Create the objective functiif items on defined in SymPy syntax
        """
        # xs = [f"x{i}" for i in range(len(self.max_cut_graph.nodes))]
        equation: Expr = cast(Expr, 0)
        for u, v in self.max_cut_graph.edges:
            u_var, v_var = self.variables[u-1], self.variables[v-1]
            equation += u_var + v_var -2*u_var*v_var
        equation *= -1
        
        self.objective_function = Expression(equation)



class BrainCommunityDetection(Problem):
    def __init__():
        pass
        

