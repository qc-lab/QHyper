import itertools

import pennylane as qml
from pennylane import numpy as np

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
        number_of_cities, 
        number_of_layers: int = 6, 
        optimization_steps: int = 70,
        optimizer: qml.GradientDescentOptimizer = qml.AdagradOptimizer()
    ):
        self.tsp_instance = TSP(number_of_cities)
        self.number_of_layers = number_of_layers
        self.wires = number_of_cities**2
        self.optimization_steps = optimization_steps
        self.optimizer = optimizer
        self.dev = qml.device("default.qubit", wires=self.wires)
    
    def _calc_bit(self, i, t):
        return i + t * self.tsp_instance.number_of_cities

    def _x(self, i, t):
        wire = self._calc_bit(i, t)
        return qml.Hamiltonian([0.5, -0.5], [qml.Identity(wire), qml.PauliZ(wire)])

    def create_cost_operator(self, parameters: list[float]):
        A_1, A_2, B = parameters
        
        cost_of_constraint_each_visited = 0    
        for i in range(self.tsp_instance.number_of_cities):
            curr = qml.Identity(0)
            for t in range(self.tsp_instance.number_of_cities):
                curr -= self._x(i, t)    
            for t1 in range(self.tsp_instance.number_of_cities):
                for t2 in range(t1 + 1, self.tsp_instance.number_of_cities):
                    curr += 2 * self._x(i, t1) @ self._x(i, t2)
            cost_of_constraint_each_visited += curr
        cost_of_constraint_each_visited_once = 0
        for t in range(self.tsp_instance.number_of_cities):
            curr = qml.Identity(0)
            for i in range(self.tsp_instance.number_of_cities):
                curr -= self._x(i, t)
            for i1 in range(self.tsp_instance.number_of_cities):
                for i2 in range(i1 + 1, self.tsp_instance.number_of_cities):
                    curr += 2 * self._x(i1, t) @ self._x(i2, t)
            cost_of_constraint_each_visited += curr
        
        cost_of_visiting_cities = 0
        for i, j in itertools.permutations(range(0, self.tsp_instance.number_of_cities), 2):
            curr = qml.Identity(0)
            for t in range(self.tsp_instance.number_of_cities):
                inc_t = t + 1
                if inc_t == self.tsp_instance.number_of_cities:
                    inc_t = 0
                curr += self._x(i, t) @ self._x(j, inc_t)
            cost_of_visiting_cities += float(self.tsp_instance.normalized_distance_matrix[i][j]) * curr 
        
        cost_operator = (
            A_1 * cost_of_constraint_each_visited + 
            A_2 * cost_of_constraint_each_visited_once +
            B * cost_of_visiting_cities
        )
                
        return cost_operator
    
    def _get_distance(self, key):
        get_bin = lambda x: format(x, 'b').zfill(self.wires)
        results = np.array_split(list(get_bin(key)), self.tsp_instance.number_of_cities)
        dist = 0
        tab = []
        for result in results:
            tab.append(list(result).index('1'))

        for i in range(len(tab)):
            j = i - 1
            # print(tsp_instance.normalized_distance_matrix[i][j])
            dist += self.tsp_instance.normalized_distance_matrix[tab[i]][tab[j]]

        return dist

    def check_results(self, probs):
        get_bin = lambda x: format(x, 'b').zfill(self.wires)
        correct_results_3 = ("100010001", "100001010", "010100001", "010001100", "001100010", "001010100")
        correct_results_4 = ("0001100001000010","0010010010000001","0100100000010010","1000000100100100","1000010000100001","0100001000011000","0001001001001000","0010000110000100","0100000110000010","0010100000010100","0001010000101000","0001100000100100","1000000101000010","1000001001000001","0100001010000001", "0100000100101000", "0010010000011000", "0100100000100001", "1000001000010100", "0001001010000100", "0001010010000010","0010000101001000", "1000010000010010", "0010100001000001")
        
        if self.tsp_instance.number_of_cities == 2:
            correct_results = ("1001", "0110")
        elif self.tsp_instance.number_of_cities == 3:
            correct_results = correct_results_3
        elif self.tsp_instance.number_of_cities == 4:
            correct_results = correct_results_4
        else:
            raise Exception(f"Provide correct results for {self.tsp_instance.number_of_cities} cites")

        result_dict = {key: float(val) for key, val in enumerate(probs)}
        result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
        score = 0
        for key, val in result_dict.items():
            if get_bin(key) not in correct_results:
                score += val*20
            else:
                score += val*self._get_distance(key)
        return score
    
    def get_score(self, result):
        pass
