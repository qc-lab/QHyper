import itertools

import numpy as np
import sympy

from sympy.core.expr import Expr
from typing import cast

from .base import Problem
from ..hyperparameter_gen.parser import Expression


class TSP:
    """Traveling Salesman Problem

    Attributes
    ----------
    number_of_cities : int
        cities count
    coords_range : tuple[int, int]
        range of coordinates of the cities
    cities_coords : list[tuple[float, float]]
        coordinates of the cities
    distance_matrix : list[list[float]]
        matrix of distances between cities
    normalized_distance_matrix : list[list[float]]
        normalized to (0, 1] matrix of distances between cities
    """

    def __init__(self, number_of_cities: int, coords_range: tuple[int, int] = (0, 10000)) -> None:
        """
        Parameters
        ----------
        number_of_cities : int
            cities count
        coords_range : tuple[float, float]
            range of coordinates of the cities
        """
        self.number_of_cities: int = number_of_cities
        self.coords_range: tuple[int, int] = coords_range
        self.cities_coords: list[tuple[float, float]] = self.get_cities()
        self.distance_matrix: list[list[float]] = self.calculate_distance_matrix()
        self.normalized_distance_matrix: list[list[float]] = self.normalize_distance_matrix()

    def get_cities(self) -> list[tuple[float, float]]:
        cities_coords = np.random.randint(self.coords_range[0], self.coords_range[1], size=(self.number_of_cities, 2))
        return cast(list[tuple[float, float]], cities_coords)

    def calculate_distance_between_points(
            self, point_A: tuple[float, float], point_B: tuple[float, float]) -> float:
        return cast(float, np.sqrt((point_A[0] - point_B[0]) ** 2 + (point_A[1] - point_B[1]) ** 2))

    def calculate_distance_matrix(self) -> list[list[float]]:
        distance_matrix = np.zeros((self.number_of_cities, self.number_of_cities))
        for i in range(self.number_of_cities):
            for j in range(i, self.number_of_cities):
                distance_matrix[i][j] = self.calculate_distance_between_points(
                    self.cities_coords[i], self.cities_coords[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        return cast(list[list[float]], distance_matrix)

    def normalize_distance_matrix(self) -> list[list[float]]:
        return cast(list[list[float]], np.divide(self.distance_matrix, np.max(self.distance_matrix)))


class TSPProblem(Problem):
    """Class defining objective function and constraints for TSP problem
    
    Attributes
    ----------
    objective_function : str
        objective function in SymPy syntax
    constraints : list[str]
        list of constraints in SymPy syntax
    variables : int
        number of qubits in the circuit, equals to number of cities to the power of 2
    """

    def __init__(
            self,
            number_of_cities: int
    ) -> None:
        """
        Parameters
        ----------
        number_of_cities : int
            number of cities
        """

        self.tsp_instance = TSP(number_of_cities)
        self.variables = sympy.symbols(' '.join([f'x{i}' for i in range(number_of_cities ** 2)]))
        self._set_objective_function()
        self._set_constraints()

    def _calc_bit(self, i: int, t: int) -> int:
        return i + t * self.tsp_instance.number_of_cities

    def _set_objective_function(self) -> None:
        equation: Expr = cast(Expr, 0)
        for i, j in itertools.permutations(range(0, self.tsp_instance.number_of_cities), 2):
            curr = 0
            for t in range(self.tsp_instance.number_of_cities):
                inc_t = t + 1
                if inc_t == self.tsp_instance.number_of_cities:
                    inc_t = 0
                curr += self.variables[self._calc_bit(i, t)] * self.variables[self._calc_bit(j, inc_t)]
            equation += self.tsp_instance.normalized_distance_matrix[i][j] * curr

        self.objective_function = Expression(equation)

    def _set_constraints(self) -> None:
        self.constraints: list[Expression] = []
        for i in range(self.tsp_instance.number_of_cities):
            equation: Expr = cast(Expr, 1)
            for t in range(self.tsp_instance.number_of_cities):
                equation -= self.variables[self._calc_bit(i, t)]
            self.constraints.append(Expression(equation))

        for t in range(self.tsp_instance.number_of_cities):
            equation = cast(Expr, 1)
            for i in range(self.tsp_instance.number_of_cities):
                equation -= self.variables[self._calc_bit(i, t)]
        self.constraints.append(Expression(equation))

    def _get_distance(self, key: str) -> float:
        results = np.array_split(list(key), self.tsp_instance.number_of_cities)
        dist: float = 0
        tab = []
        for result in results:
            tab.append(list(result).index('1'))

        for i in range(len(tab)):
            j = i - 1
            dist += self.tsp_instance.normalized_distance_matrix[tab[i]][tab[j]]

        return dist

    def _valid(self, result: str) -> bool:
        result_bools = np.reshape(list(result), (-1, self.tsp_instance.number_of_cities)).astype(np.bool8)
        return cast(bool, (result_bools.sum(0) == 1).all() and (result_bools.sum(1) == 1).all())

    def get_score(self, result: str) -> float | None:
        """Returns length of the route based on provided outcome in bits. 
        
        Parameters
        ----------
        result : str
            route as a string of zeros and ones

        Returns
        -------
        float | None
            Returns length of the route, or None if route wasn't correct
        """
        if not self._valid(result):
            return None  # Bigger value that possible distance
        return self._get_distance(result)
