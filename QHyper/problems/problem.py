from abc import ABC

import dimod

from QHyper.hyperparameter_gen.parser import Polynomial, dict_to_list


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
    wires : int
        number of qubits in the circuit

    """
    objective_function: Polynomial
    constraints: dict[Polynomial]
    wires: int
    variables: list[str]  # todo when you have that, you don't need wires --> (len(variables))

    def to_cqm(self):
        binary_polynomial = dimod.BinaryPolynomial(self.objective_function.as_dict(), dimod.BINARY)
        cqm = dimod.make_quadratic_cqm(binary_polynomial)

        for sense, constraints in self.constraints.items():
            for constraint in constraints:
                constraint = constraint.as_dict()
                constraint = dict_to_list(constraint)  # todo check what is wrong with adding dicts
                cqm.add_constraint(constraint, sense)

        return cqm

    def to_qubo(self, method="dimod"):  # method = cem
        cqm = self.to_cqm()
        bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10)
        return bqm.to_qubo()  # (qubo, offset)

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
