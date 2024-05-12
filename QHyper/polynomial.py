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
        dictionary of terms and their coefficients

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

    def __pow__(self, power: int) -> 'Polynomial':
        if not isinstance(power, int):
            raise TypeError(f"Unsupported operation: {self} ** {power}")

        if power == 0:
            return Polynomial({tuple(): 1})

        result = self
        for _ in range(power - 1):
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
        constant = self.terms.pop(tuple(), 0)
        return Polynomial(self.terms), constant

    def degree(self) -> int:
        return max(len(term) for term in self.terms)

    def get_variables(self) -> set[str]:
        return set(variable for term in self.terms for variable in term)
