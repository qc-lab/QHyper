from abc import ABC, abstractmethod

# import pennylane as qml
# from pennylane import numpy as np

from qiskit.algorithms.optimizers import ESCH


class Problem(ABC):
    objective_function: str
    constraints: list[str]
    wires: int
    # dev: qml.QNode
    # optimization_steps: int
    # number_of_layers: int
    # optimizer: qml.GradientDescentOptimizer

    # @abstractmethod
    # def create_cost_operator(self, hyperparameters) -> qml.Hamiltonian: # qubo, cost function
    #     pass

    # @abstractmethod
    # def get_score(self, result):
    #     pass
    
    # def _hadamard_layer(self):
    #     for i in range(self.wires):
    #         qml.Hadamard(i)
    
    # def create_mixing_hamitonian(self, const=1/2):
    #     hamiltonian = qml.Hamiltonian([], [])
    #     for i in range(self.wires):
    #         hamiltonian += qml.Hamiltonian([const], [qml.PauliX(i)])
    #     return hamiltonian
    
    # def check_results(self, probs):
    #     to_bin = lambda x: format(x, 'b').zfill(self.wires)
        
    #     results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
    #     results_by_probabilites = dict(
    #         sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
    #     score = 0
    #     for result, prob in results_by_probabilites.items():
    #         if (value:=self.get_score(to_bin(result))) == -1:
    #             score += 0 # experiments?
    #         else:
    #             score -= prob*value
    #     return score 
    
    # def print_results(self, probs):
    #     to_bin = lambda x: format(x, 'b').zfill(self.wires)
    #     results_by_probabilites = {result: float(prob) for result, prob in enumerate(probs)}
    #     results_by_probabilites = dict(
    #         sorted(results_by_probabilites.items(), key=lambda item: item[1], reverse=True))
    #     for result, prob in results_by_probabilites.items():
    #         # binary_rep = to_bin(key)
    #         value = self.get_score(to_bin(result))
    #         print(
    #             f"Key: {to_bin(result)} with probability {prob}   "
    #             f"| correct: {'True, value: '+str(value) if value != -1 else 'False'}"
    #         )
        
    # def circuit(self, params, cost_operator, layers):
    #     def qaoa_layer(gamma, beta):
    #         qml.qaoa.cost_layer(gamma, cost_operator)
    #         qml.qaoa.mixer_layer(beta, self.create_mixing_hamitonian()) 

    #     self._hadamard_layer()
    #     qml.layer(qaoa_layer, layers, params[0], params[1]) 

    # def get_expval_func(self, weights, layers):
    #     cost_operator = self.create_cost_operator(weights)
    #     @qml.qnode(self.dev)
    #     def cost_function(params):
    #         self.circuit(params, cost_operator, layers)
    #         x = qml.expval(cost_operator)
    #         return x
        
    #     return cost_function

    # def get_probs_func(self, weights, layers):
    #     cost_operator = self.create_cost_operator(weights)
    #     @qml.qnode(self.dev)
    #     def probability_circuit(params):
    #         self.circuit(params, cost_operator, layers)
    #         return qml.probs(wires=range(self.wires))

    #     return probability_circuit

    
    # def _hadamard_layer(self):
    #     for i in range(self.wires):
    #         qml.Hadamard(i)

    # def _run_learning(self, parameters: list[float], cost_operator: qml.Hamiltonian = None):
    #     cost_operator = cost_operator if cost_operator else self._create_cost_operator(parameters)
    #     # esch = ESCH(max_evals=200)

    #     def qaoa_layer(gamma, beta):
    #         qml.qaoa.cost_layer(gamma, cost_operator)
    #         qml.qaoa.mixer_layer(beta, self._create_mixing_hamitonian())
        
    #     def circuit(params):
    #         self._hadamard_layer()
    #         qml.layer(qaoa_layer, self.number_of_layers, params[0], params[1])    

    #         # params_0 = [param for param in params[:len(params)//2]]
    #         # params_1 = [param for param in params[len(params)//2:]]
    #         # qml.layer(qaoa_layer, self.number_of_layers, params_0, params_1)    

    #     @qml.qnode(self.dev)
    #     def cost_function(params):
    #         circuit(params)
    #         x = qml.expval(cost_operator)
    #         return x
        
    #     @qml.qnode(self.dev)
    #     def probability_circuit(params):
    #         circuit(params)
    #         return qml.probs(wires=range(self.wires))

    #     def wrapper(params):
    #         probs = probability_circuit(params)
    #         results = float(self._check_results(probs))
    #         return results

    #     steps = self.optimization_steps
    #     params = np.array(
    #         [[0.5]*self.number_of_layers, [0.5]*self.number_of_layers], 
    #         requires_grad=True
    #     )
        
    #     for _ in range(steps):
    #         params = self.optimizer.step(cost_function, params)


    #     # params = np.array(
    #     #     [[0.5]*2*self.number_of_layers], 
    #     #     requires_grad=True
    #     # )

    #     # result = esch.minimize(wrapper, params[0])
    #     # params = result.x

    #     probs = probability_circuit(params)
    #     return probs


    # def run_learning_n_get_results(self, p):
    #     p = [float(x) for x in p]
    #     probs = self._run_learning(p)
    #     return float(self._check_results(probs))
