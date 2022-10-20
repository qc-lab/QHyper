from abc import ABC

from qiskit.algorithms.optimizers import ESCH


class Problem(ABC):
    """Base class for different problems
    
    Objective function and constrians are main components, which should be written in SymPy syntax.
    Depending on solver, which will be used, these parts can be used separately or e.g as a QUBO.
    
    If QUBO is provided, should be pass to objective_function, and leave constrains empty.

    Args:
        - objective_function - a string representing objective function in SymPy syntax
        - constraints - list of strings, each in SymPy syntax
        - wires - an integer indicating how many wires should be in the circuit

    """
    objective_function: str
    constraints: list[str]
    wires: int
 