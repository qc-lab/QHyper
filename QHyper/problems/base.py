from abc import ABC

from QHyper.hyperparameter_gen.parser import Expression


class Problem(ABC):
    """Interface for different combinatorial optimization problems

    Objective function and constrians are the main components and should be written in the SymPy syntax.
    Depending on the selcted solver, these parts can be used separately or, e.g., as a Quadratic Unconstrained
    Binary Optimization (QUBO) formularion.

    If the QUBO is provided, it should be passed to the objective_function and the constraints should be empty.

    Attributes
    ----------
    objective_function
        objective function in SymPy syntax
    constraints : list[str]
        list of constraints in SymPy syntax
    variables : int
        number of variables in the problem
    name: str
        helps to differentiate various problems (default "")
    """
    objective_function: Expression
    constraints: list[Expression]
    variables: list
    name: str = ""

    def get_score(self, result: str) -> float | None:
        """Returns score of the outcome provided as a binary string

        Necessary only for hyperoptimizers. Not needed for minimizing functions with constant weights.

        Parameters
        ----------
        result : str
            outcome as a string of zeros and ones

        Returns
        -------
        float | None
            Returns float if outcome is correct and meets all criteria, otherwise returns None
        """
        raise Exception("Unimplemented")

    def __repr__(self) -> str:
        if self.name == "":
            return super().__repr__()
        return f"{self.__class__.__name__}_{self.name}"