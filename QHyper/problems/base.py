# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from abc import ABC
import numpy as np

from QHyper.constraint import Constraint, Polynomial


class ProblemException(Exception):
    pass


class Problem(ABC):
    """Interface for different combinatorial optimization problems

    Objective function and constrians are the main components
    and are represented as :py:class:`~QHyper.polynomial.Polynomial`.
    Depending on the selcted solver, these parts can be used
    separately or, e.g., as a Quadratic Unconstrained
    Binary Optimization (QUBO) formularion.

    If the QUBO is provided, it should be passed to the
    objective_function and the constraints should be empty.
    Same applies for the situation when the problem doesn't
    have constraints.

    Attributes
    ----------
    objective_function: Polynomial
        Objective_function represented as a 
        :py:class:`~QHyper.polynomial.Polynomial`
    constraints : list[Polynomial], optional
        List of constraints represented as a 
        :py:class:`~QHyper.polynomial.Polynomial`
    """

    objective_function: Polynomial
    constraints: list[Constraint] = []

    def get_score(self, result: np.record, penalty: float = 0) -> float:
        """Returns score of the outcome provided as a binary string

        Necessary to evaluate results. It's not possible to calculate the
        score based on the objective function directly, because that is
        highly dependent on the problem. That's why this method requires
        user to implement it.
        This method is used in optimization process where evaluating QUBO
        or expectation value is not enough.

        Parameters
        ----------
        result : np.record
            Outcome as a numpy record with variables as keys and their values.
            Dtype is list of tuples with variable name and its value (0 or 1)
            and tuple ('probability', <float>).
        penalty : float, default 0
            Penalty for the constraint violation

        Returns
        -------
        float
            Returns float indicating the score, if function should be
            maximized the returned value should be returned with negative sign
        """
        raise NotImplementedError("Unimplemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
