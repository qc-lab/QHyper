from abc import ABC


class Problem(ABC):
    """Interface for different combinatorial optimization problems
    
    Objective function and constrians are main the components and should be written in the SymPy syntax.
    Depending on the selcted solver, these parts can be used separately or, e.g., as a Quadratic Unconstrained
    Binary Optimization (QUBO) formularion.
    
    If the QUBO is provided, it should be passed to the objective_function and the constraints should be empty.

    Attributes
    ----------
    objective_function 
        objective function in SymPy syntax
    constraints : list[str]
        list of constraints in SymPy syntax
    wires : int
        number of qubits in the circuit

    """
    objective_function: str
    constraints: list[str]
    wires: int

    def get_score(self, result: str) -> float | None:
        """Returns score of the outcome provided as a binary string

        Necessary only for specific functions. Not needed for minimizing the function.
        
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
