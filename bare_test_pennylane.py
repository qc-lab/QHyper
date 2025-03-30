import pennylane as qml
import networkx as nx
import time
from pennylane import numpy as np

# edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 9), (0, 14), (1, 2), (1, 3), (1, 5), (1, 9), (1, 10), (1, 11), (1, 12), (1, 14), (2, 5), (2, 9), (2, 12), (2, 14), (3, 4), (3, 7), (3, 8), (3, 9), (3, 10), (4, 6), (4, 7), (4, 10), (4, 12), (4, 13), (5, 9), (5, 10), (6, 8), (6, 12), (7, 9), (7, 10), (8, 10), (8, 14), (9, 10), (9, 11), (11, 12), (11, 13), (12, 14)]
edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 1), (3, 5)]
graph = nx.Graph(edges)


def qaoa_layer(gamma, beta):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, mixer_h)


def circuit(params):
    for n in graph.nodes:
        qml.Hadamard(wires=n)
    qml.layer(qaoa_layer, layers, params[0], params[1])


if __name__ == '__main__':

    problem_configuration_times = []
    solver_times =[]

    for i in range(10):
        start_time_problem = time.perf_counter()
        # identity_h = qml.Hamiltonian(
        # [-0.5 for e in graph.edges],
        # [qml.Identity(e[0]) @ qml.Identity(e[1]) for e in graph.edges],
        # )
        # zz_h = qml.Hamiltonian(
        #     [0.5 for e in graph.edges],
        #     [qml.Z(e[0]) @ qml.Z(e[1]) for e in graph.edges]
        # )
        # cost_h= zz_h + identity_h
        cost_h, mixer_h = qml.qaoa.maxcut(graph)
        problem_configuration_times.append(time.perf_counter() - start_time_problem)

        # mixer_h = qml.qaoa.x_mixer(graph.nodes)

        layers = 2
        params = qml.numpy.array([[0.5, 0.5], [1.0, 1.0]], requires_grad=True)

        dev = qml.device("default.qubit", wires=range(len(graph.nodes)))

        @qml.qnode(dev)
        def cost_function(params):
            circuit(params)
            return qml.expval(cost_h.simplify())

        start_time_solver = time.perf_counter()
        step_size = 0.005
        steps = 70
        optimizer = qml.AdamOptimizer(stepsize=step_size)
        time_ = time.perf_counter()
        for i in range(steps):
            params = optimizer.step(cost_function, params)
            # params = np.array(params, requires_grad=True)
        print("Time: ", time.perf_counter() - time_)

        @qml.qnode(dev)
        def probability_circuit(gamma, beta):
            circuit([gamma, beta])
            return qml.probs(wires=graph.nodes)

        probs = probability_circuit(params[0], params[1])
        solver_times.append(time.perf_counter() - start_time_solver)

    print("Problem configuration times: ", problem_configuration_times)
    print("Solver times: ", solver_times)
    print("Average problem configuration time: ", sum(problem_configuration_times[1:])/(len(problem_configuration_times)-1))
    print("Average solver time: ", sum(solver_times[1:])/(len(solver_times)-1))
