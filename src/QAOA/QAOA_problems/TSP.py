import itertools

import numpy as np

from .problem import Problem


class TSP:
    def __init__(self, number_of_cities, coords_range=(0, 10000)):
        self.number_of_cities = number_of_cities
        self.coords_range = coords_range
        self.cities_coords = self.get_cities()
        self.distance_matrix = self.calculate_distance_matrix()
        self.normalized_distance_matrix = self.normalize_distance_matrix()
    
    def get_cities(self):
        cities_coords = np.random.randint(self.coords_range[0], self.coords_range[1], size = (self.number_of_cities, 2))
        return cities_coords
           
    def normalize_cities(self):
        max_coords = np.amax(self.cities_coords, axis=0)
        normalized_cities_coords = np.divide(self.cities_coords, max_coords)
        return normalized_cities_coords

    def calculate_distance_between_points(self, point_A, point_B):
        return np.sqrt((point_A[0] - point_B[0])**2 + (point_A[1] - point_B[1])**2)
    
    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.number_of_cities, self.number_of_cities))
        for i in range(self.number_of_cities):
            for j in range(i, self.number_of_cities):
                distance_matrix[i][j] = self.calculate_distance_between_points(self.cities_coords[i], self.cities_coords[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix 
    
    def normalize_distance_matrix(self):
        return np.divide(self.distance_matrix, np.max(self.distance_matrix))


class QAOA_TSP(Problem):
    def __init__(
        self, 
        number_of_cities
    ):
        self.tsp_instance = TSP(number_of_cities)
        self.wires = number_of_cities**2
        self.create_objective_function()
        self.create_constraints()

    def _calc_bit(self, i, t):
        return i + t * self.tsp_instance.number_of_cities
    
    def create_objective_function(self):
        equation = ""
        for i, j in itertools.permutations(range(0, self.tsp_instance.number_of_cities), 2):
            equation += f"+{self.tsp_instance.normalized_distance_matrix[i][j]}*("
            for t in range(self.tsp_instance.number_of_cities):
                equation += f"+x{self._calc_bit(i, t)}*x{self._calc_bit(j, (t+1)%self.tsp_instance.number_of_cities)}"
            equation += ")"
        
        self.objective_function = equation

    def create_constraints(self):
        self.constraints = []
        equation = ""
        for i in range(self.tsp_instance.number_of_cities):
            equation += f"+(J"
            for t in range(self.tsp_instance.number_of_cities):
                equation += f"-x{self._calc_bit(i, t)}"
            equation += f")**2"
        
        
        self.constraints.append(equation)
        equation = ""
        for t in range(self.tsp_instance.number_of_cities):
            equation += f"+(J"
            for i in range(self.tsp_instance.number_of_cities):
                equation += f"-x{self._calc_bit(i, t)}"
            equation += f")**2"
        self.constraints.append(equation)
    
    def _get_distance(self, key):
        results = np.array_split(list(key), self.tsp_instance.number_of_cities)
        dist = 0
        tab = []
        for result in results:
            tab.append(list(result).index('1'))

        for i in range(len(tab)):
            j = i - 1
            dist += self.tsp_instance.normalized_distance_matrix[tab[i]][tab[j]]

        return dist

    def _valid(self, result):
        result = np.reshape(list(result), (-1, self.tsp_instance.number_of_cities)).astype(np.bool8)
        return (result.sum(0) == 1).all() and (result.sum(1) == 1).all()

    def get_score(self, result) -> float | None:
        if not self._valid(result):
            return None # Bigger value that possible distance 
        return self._get_distance(result)
