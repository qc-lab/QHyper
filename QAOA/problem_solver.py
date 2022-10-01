import pennylane as qml
from pennylane import numpy as np

from .problems.problem import Problem


class ProblemSolver:
    problem: Problem
    optimization_steps: int
    dev: qml.QNode
    optimizer: qml.GradientDescentOptimizer
    layers: int

    def __init__(
        self, 
        problem: Problem, 
        optimizer: qml.GradientDescentOptimizer,
        optimization_steps: int = 200,
        layers: int = 1
    ) -> None:
        self.problem = problem
        self.dev = qml.device("default.qubit", wires=problem.wires)
        self.optimizer = optimizer
        self.optimization_steps = optimization_steps
        self.layers = layers
    
    def _hadamard_layer(self):
        for i in range(self.problem.wires):
            qml.Hadamard(i)

    def run_learning(
        self, hyperparameters: list[float], 
        cost_operator: qml.Hamiltonian = None
    ):
        # hyperparameters = [float(x) for x in hyperparameters]
        cost_operator = cost_operator if cost_operator else self.problem.create_cost_operator(hyperparameters)
        # esch = ESCH(max_evals=200)

        def qaoa_layer(gamma, beta):
            qml.qaoa.cost_layer(gamma, cost_operator)
            qml.qaoa.mixer_layer(beta, self.problem.create_mixing_hamitonian())
        
        def circuit(params):
            self._hadamard_layer()
            qml.layer(qaoa_layer, self.layers, params[0], params[1])    

            # params_0 = [param for param in params[:len(params)//2]]
            # params_1 = [param for param in params[len(params)//2:]]
            # qml.layer(qaoa_layer, self.number_of_layers, params_0, params_1)    

        @qml.qnode(self.dev)
        def cost_function(params):
            circuit(params)
            x = qml.expval(cost_operator)
            return x
        
        @qml.qnode(self.dev)
        def probability_circuit(params):
            circuit(params)
            return qml.probs(wires=range(self.problem.wires))

        # def wrapper(params):
        #     probs = probability_circuit(params)
        #     results = float(self._check_results(probs))
        #     return results

        params = np.array(
            [[0.5]*self.layers, [0.5]*self.layers], 
            requires_grad=True
        )
        # print(cost_operator)
        # print(cost_function(params))
        for _ in range(self.optimization_steps):
            params = self.optimizer.step(cost_function, params)


        # # params = np.array(
        # #     [[0.5]*2*self.number_of_layers], 
        # #     requires_grad=True
        # # )

        # # result = esch.minimize(wrapper, params[0])
        # # params = result.x

        probs = probability_circuit(params)
        return params, probs

    def get_score(self, hyperparameters: list[float]):
        _, probs = self.run_learning(hyperparameters)
        return self.problem.check_results(probs)

    # def run_learning_n_get_results(self, p):
    #     p = [float(x) for x in p]
    #     probs = self._run_learning(p)
    #     return float(self._check_results(probs))
