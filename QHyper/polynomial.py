""" Module for polynomial representation.
Implementation of the polynomials using dictionaries. Used in the whole system.

.. rubric:: Main class

.. autosummary::
    :toctree: generated

    Polynomial  -- implementation of the polynomial.


.. rubric:: MyPy Type

.. autoclass:: PolynomialType

"""

from dataclasses import dataclass, field
from collections import defaultdict

from typing import overload


@dataclass
class Polynomial:
    """
    Class for representing polynomials.

    A Polynomial is comprised of a dictionary where the keys are tuples
    containing variables, and the values represent their coefficients.
    Using dictionaries allows for efficient arithmetic operations
    on Polynomials, simplification of terms, and extraction of relevant
    information such as constants, degree, and variables.
    The core functionality includes addition, subtraction,
    multiplication, exponentiation, and negation. This representation
    can store higher-order polynomials. The creation of a Polynomial
    can be done manually by providing a dictionary or by translating
    it from SymPy syntax. Additionally, users can implement the
    translation into Polynomial from their own data source

    Attributes
    ----------
    terms : dict[tuple[str, ...], float]
        dictionary of terms and their coefficients, where the key is a tuple
        of variables and the value is the coefficient of the term
        For example, the polynomial 3*x + 2 + 4*x^2 is represented as
        {('x',): 3, ('x', 'x'): 4, (): 2}
    """

    terms: dict[tuple[str, ...], float] = field(default_factory=dict)

    @overload
    def __init__(self, terms: float | int) -> None: ...

    @overload
    def __init__(self, terms: dict[tuple[str, ...], float]) -> None: ...

    def __init__(self, terms: dict[tuple[str, ...], float] | float | int
                 ) -> None:
        if isinstance(terms, (float, int)):
            terms = {tuple(): float(terms)}
        else:
            terms = terms.copy() if terms else {tuple(): 0}

        self.terms = defaultdict(float)

        for term, coefficient in terms.items():
            if coefficient == 0:
                continue
            self.terms[tuple(sorted(term))] += coefficient

    @overload
    def __add__(self, other: float | int) -> 'Polynomial': ...

    @overload
    def __add__(self, other: 'Polynomial') -> 'Polynomial': ...

    def __add__(self, other: 'Polynomial | float | int') -> 'Polynomial':
        if isinstance(other, (float, int)):
            return Polynomial({tuple(): float(other)}) + self

        if not isinstance(other, Polynomial):
            raise TypeError(f"Unsupported operation: {self} + {other}")

        new_terms = self.terms.copy()

        for term, coefficient in other.terms.items():
            new_terms[term] += coefficient

        return Polynomial(new_terms)

    def __radd__(self, other: float | int) -> 'Polynomial':
        return self + other

    @overload
    def __sub__(self, other: float | int) -> 'Polynomial': ...

    @overload
    def __sub__(self, other: 'Polynomial') -> 'Polynomial': ...

    def __sub__(self, other: 'Polynomial | float | int') -> 'Polynomial':
        if isinstance(other, (float, int)):
            return Polynomial({tuple(): float(other)}) - self
        if not isinstance(other, Polynomial):
            raise TypeError(f"Unsupported operation: {self} - {other}")

        new_terms = self.terms.copy()

        for term, coefficient in other.terms.items():
            new_terms[term] -= coefficient

        return Polynomial(new_terms)

    def __rsub__(self, other: float | int) -> 'Polynomial':
        return -self + other

    @overload
    def __mul__(self, other: float | int) -> 'Polynomial': ...

    @overload
    def __mul__(self, other: 'Polynomial') -> 'Polynomial': ...

    def __mul__(self, other: 'Polynomial | float | int') -> 'Polynomial':
        if isinstance(other, (float, int)):
            return Polynomial({tuple(): float(other)}) * self
        if not isinstance(other, Polynomial):
            raise TypeError(f"Unsupported operation: {self} * {other}")

        new_terms = defaultdict(float)

        for variables1, coefficient1 in self.terms.items():
            for variables2, coefficient2 in other.terms.items():
                new_term = tuple(sorted(variables1 + variables2))
                new_coefficient = coefficient1 * coefficient2

                new_terms[new_term] += new_coefficient

        return Polynomial(new_terms)

    def __rmul__(self, other: float | int) -> 'Polynomial':
        return self * other

    def __pow__(self, power: 'Polynomial | int') -> 'Polynomial':
        power_: int

        if isinstance(power, Polynomial):
            const = power.terms.get(tuple(), 0)
            if power.degree() != 0 or (int(const) != const):
                raise ValueError(f"Unsupported operation: {self} ** {power}")
            power_ = int(const)
        elif not isinstance(power, int):
            raise TypeError(f"Unsupported operation: {self} ** {power}")
        else:
            power_ = power
        if power_ == 0:
            return Polynomial({tuple(): 1})

        result = self
        for _ in range(power_ - 1):
            result *= self

        return result

    def __neg__(self) -> 'Polynomial':
        return Polynomial({
            term: -coefficient for term, coefficient in self.terms.items()
        })

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            terms = other
        elif isinstance(other, Polynomial):
            terms = other.terms
        else:
            raise TypeError(f"Unsupported operation: {self} == {other}")

        return self.terms == terms

    def separate_const(self) -> tuple['Polynomial', float]:
        """Method for separating constant term from the rest of the polynomial.

        For example, for polynomial 3*x + 2 + 4*x^2, the method will return
        3*x + 4*x^2 and 2 or in the actual representation:
        {('x',): 3, ('x', 'x'): 4, (): 2} will be separated into
        {('x',): 3, ('x', 'x'): 4} and {(): 2}.

        Returns
        -------
        Polynomial
            Polynomial without the constant term - empty tuple.
        float
            Constant term of the polynomial.
        """

        _terms = self.terms.copy()
        constant = _terms.pop(tuple(), 0)
        return Polynomial(_terms), constant

    def degree(self) -> int:
        """Method for calculating the degree of the polynomial.

        Returns
        -------
        int
            The degree of the polynomial.
        """
        return max(len(term) for term in self.terms)

    def get_variables(self) -> set[str]:
        """Method for extracting variables from the polynomial.

        Returns
        -------
        set[str]
            Set of variables used in the polynomial.
        """
        return set(variable for term in self.terms for variable in term)


PolynomialType = Polynomial | float | int | dict[tuple[str, ...], float]
