from dataclasses import dataclass, field
from collections import defaultdict

from typing import overload

import sympy


@dataclass
class Polynomial:
    terms: dict[tuple[str, ...], float] = field(default_factory=dict)

    def __init__(self, terms: dict[tuple[str, ...], float] = {tuple(): 0}) -> None:
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

    @overload
    def __mul__(self, other: float | int) -> 'Polynomial': ...

    @overload
    def __mul__(self, other: 'Polynomial') -> 'Polynomial': ...

    def __mul__(self, other: 'Polynomial | float | int') -> 'Polynomial':
        if isinstance(other, (float, int)):
            return Polynomial({tuple(): float(other)}) * self
        if not isinstance(other, Polynomial):
            raise TypeError(f"Unsupported operation: {self} * {other}")

        new_terms = {}

        for variables1, coefficient1 in self.terms.items():
            for variables2, coefficient2 in other.terms.items():
                new_term = tuple(sorted(variables1 + variables2))
                new_coefficient = coefficient1 * coefficient2

                new_terms[new_term] += new_coefficient

        return Polynomial(new_terms)

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

    def separate_const(self) -> tuple['Polynomial', float]:
        constant = self.terms.pop(tuple(), 0)
        return Polynomial(self.terms), constant

    def degree(self) -> int:
        return max(len(term) for term in self.terms)

    def get_variables(self) -> set[str]:
        return set(variable for term in self.terms for variable in term)
