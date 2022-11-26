from abc import ABC


class Problem(ABC):
    """Interface for different problems
    
    Objective function and constrians are main components, which should be written in SymPy syntax.
    Depending on solver, which will be used, these parts can be used separately or e.g. as a QUBO.
    
    If QUBO is provided, should be pass to objective_function, and leave constrains empty.

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
        """Method should return score of the provided outcome in bits. 

        Necessary only for specific function. Not needed for minimizing function.
        
        Parameters
        ----------
        result : str
            outcome as a string of zeros and ones

        Returns
        -------
        float | None
            Returns float if outcome is correct and meets all criteria, else returns None
        """
        raise Exception("Unimplemented")
 