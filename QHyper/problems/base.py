# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from abc import ABC

import sympy

from QHyper.util import Expression, Constraint


class Problem(ABC):
    """Interface for different combinatorial optimization problems

    Objective function and constrians are the main components
    and should be written in the SymPy syntax.
    Depending on the selcted solver, these parts can be used
    separately or, e.g., as a Quadratic Unconstrained
    Binary Optimization (QUBO) formularion.

    If the QUBO is provided, it should be passed to the
    objective_function and the constraints should be empty.

    Attributes
    ----------
    objective_function: Expression
        objective function in SymPy syntax
    constraints : list[Expression]
        list of constraints in SymPy syntax
    variables : list[Any]
        list of variables in the problem
    name: str
        helps to differentiate various problems (default "")
    cases: int
        number of variable cases (values)
        (default 1 - denoting binary variable)
    """

    objective_function: Expression
    constraints: list[Constraint]
    variables: tuple[sympy.Symbol]
    name: str = ""
    cases: int = 1

    def get_score(self, result: str, penalty: float = 0) -> float:
        """Returns score of the outcome provided as a binary string

        Necessary to evaluate results.

        Parameters
        ----------
        result : str
            outcome as a string of zeros and ones

        Returns
        -------
        float
            Returns float indicating the score, if function should be
            maximized the returned value should be returned with negative sign
        """
        raise Exception("Unimplemented")

    def __repr__(self) -> str:
        if self.name == "":
            return super().__repr__()
        return f"{self.__class__.__name__}_{self.name}"
