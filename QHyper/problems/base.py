# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from abc import ABC

from QHyper.constraint import Constraint, Polynomial


class ProblemException(Exception):
    pass


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
    cases: int
        number of variable cases (values)
        (default 1 - denoting binary variable)
    """

    objective_function: Polynomial
    constraints: list[Constraint] = []
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
        raise NotImplementedError("Unimplemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
