# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import itertools

import sympy
import numpy as np

from typing import cast
from QHyper.constraint import Constraint

from QHyper.parser import from_sympy
from QHyper.polynomial import Polynomial
from QHyper.problems.base import Problem


class TSP:
    """Traveling Salesman Problem

    Attributes
    ----------
    number_of_cities : int
        cities count
    cities_coords : list[tuple[float, float]]
        coordinates of the cities
    distance_matrix : list[list[float]]
        matrix of distances between cities
    normalized_distance_matrix : list[list[float]]
        normalized to (0, 1] matrix of distances between cities
    """

    def __init__(
        self,
        number_of_cities: int,
        coords_range: tuple[int, int] = (0, 10000),
        cities_coords: list[tuple[float, float]] = [],
    ) -> None:
        """
        Parameters
        ----------
        number_of_cities : int
            cities count
        coords_range : tuple[float, float]
            range of coordinates of the cities
        cities_coords : list[tuple[float, float]]
            predefined coordinates of the cities
        """
        self.number_of_cities: int = number_of_cities
        if cities_coords:
            self.cities_coords = cities_coords
        else:
            self.cities_coords = self.get_cities(coords_range)
        self.distance_matrix: list[
            list[float]] = self.calculate_distance_matrix()
        self.normalized_distance_matrix: list[
            list[float]] = self.normalize_distance_matrix()

    def get_cities(self, coords_range) -> list[tuple[float, float]]:
        cities_coords = np.random.randint(
            coords_range[0], coords_range[1],
            size=(self.number_of_cities, 2))
        return cast(list[tuple[float, float]], cities_coords)

    def calculate_distance_between_points(
        self, point_A: tuple[float, float], point_B: tuple[float, float]
    ) -> float:
        return cast(
            float,
            np.sqrt((point_A[0] - point_B[0]) ** 2
                    + (point_A[1] - point_B[1]) ** 2)
        )

    def calculate_distance_matrix(self) -> list[list[float]]:
        distance_matrix = np.zeros(
            (self.number_of_cities, self.number_of_cities))
        for i in range(self.number_of_cities):
            for j in range(i, self.number_of_cities):
                distance_matrix[i][j] = self.calculate_distance_between_points(
                    self.cities_coords[i], self.cities_coords[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        return cast(list[list[float]], distance_matrix)

    def normalize_distance_matrix(self) -> list[list[float]]:
        return cast(
            list[list[float]],
            np.divide(self.distance_matrix, np.max(self.distance_matrix))
        )


class TravelingSalesmanProblem(Problem):
    """
    Class defining objective function and constraints for TSP

    Parameters
    ----------
    number_of_cities : int
        Number of cities
    cities_coords : list[tuple[float, float]], default []
        List of cities coordinates. If not provided, random cities
        coordinates are generated.

    Attributes
    ----------
    objective_function : Polynomial
        Objective function represented as a Polynomial
    constraints : list[Polynomial]
        List of constraints represented as a Polynomials
    tsp_instance: :py:class:`TSP`
        TSP problem instace
    """

    def __init__(
        self,
        number_of_cities: int,
        cities_coords: list[tuple[float, float]] = [],
    ) -> None:
        self.tsp_instance = TSP(
            number_of_cities, cities_coords=cities_coords)
        self.variables: tuple[sympy.Symbol] = sympy.symbols(
            ' '.join([f'x{i}' for i in range(number_of_cities ** 2)]))
        self.objective_function = self._get_objective_function()
        self.constraints = self._get_constraints()

    def _calc_bit(self, i: int, t: int) -> int:
        return i + t * self.tsp_instance.number_of_cities

    def _get_objective_function(self) -> Polynomial:
        equation = Polynomial(0)
        for i, j in itertools.permutations(
            range(0, self.tsp_instance.number_of_cities), 2
        ):
            curr = Polynomial(0)
            for t in range(self.tsp_instance.number_of_cities):
                inc_t = t + 1
                if inc_t == self.tsp_instance.number_of_cities:
                    inc_t = 0
                curr += Polynomial({(f"x{self._calc_bit(i, t)}",
                                     f"x{self._calc_bit(j, inc_t)}"): 1})
            equation += (
                self.tsp_instance.normalized_distance_matrix[i][j] * curr
            )
        return equation

    def _get_constraints(self) -> list[Constraint]:
        constraints: list[Constraint] = []
        for i in range(self.tsp_instance.number_of_cities):
            equation = Polynomial(1)
            for t in range(self.tsp_instance.number_of_cities):
                equation -= Polynomial({(f"x{self._calc_bit(i, t)}",): 1})
            constraints.append(Constraint(equation, group=0))

        for t in range(self.tsp_instance.number_of_cities):
            equation = Polynomial(1)
            for i in range(self.tsp_instance.number_of_cities):
                equation -= Polynomial({(f"x{self._calc_bit(i, t)}",): 1})
            constraints.append(Constraint(equation, group=1))
        return constraints

    def _get_distance(self, order_result: np.ndarray) -> float:
        dist: float = 0
        tab = []
        for result in order_result:
            tab.append(list(result).index(1))

        for i in range(len(tab)):
            j = i - 1
            dist += (
                self.tsp_instance.normalized_distance_matrix[tab[i]][tab[j]]
            )
        return dist

    def _valid(self, order_result: np.ndarray) -> bool:
        return cast(bool, (
            order_result.sum(0) == 1).all()
            and (order_result.sum(1) == 1).all()
        )

    def get_score(self, result: np.record, penalty: float = 0) -> float:
        """Returns length of the route for provided numpy record.

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
            Returns length of the route, or 0 if route wasn't correct
        """
        order_result = np.zeros((self.tsp_instance.number_of_cities,
                                 self.tsp_instance.number_of_cities))

        for i in range(self.tsp_instance.number_of_cities):
            for j in range(self.tsp_instance.number_of_cities):
                order_result[i][j] = result[
                    f"x{i*self.tsp_instance.number_of_cities + j}"]

        if not self._valid(order_result):
            return penalty  # Bigger value that possible distance
        return self._get_distance(order_result)
