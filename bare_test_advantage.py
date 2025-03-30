import dimod
import networkx as nx
from collections import defaultdict
import time
from dwave.system import DWaveSampler, EmbeddingComposite

edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 9), (0, 14), (1, 2), (1, 3), (1, 5), (1, 9), (1, 10), (1, 11), (1, 12), (1, 14), (2, 5), (2, 9), (2, 12), (2, 14), (3, 4), (3, 7), (3, 8), (3, 9), (3, 10), (4, 6), (4, 7), (4, 10), (4, 12), (4, 13), (5, 9), (5, 10), (6, 8), (6, 12), (7, 9), (7, 10), (8, 10), (8, 14), (9, 10), (9, 11), (11, 12), (11, 13), (12, 14)]
graph = nx.Graph(edges)

if __name__ == '__main__':

    problem_configuration_times = []
    solver_times =[]

    for i in range(10):
        start_time_problem = time.perf_counter()
        linear = defaultdict(int)
        quadratic = defaultdict(int)

        for i,j in graph.edges:
            linear[i] += -1
            linear[j] += -1
            quadratic[(i,j)] += 2

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)
        problem_configuration_times.append(time.perf_counter() - start_time_problem)

        start_time_solver = time.perf_counter()
        sampler = DWaveSampler(solver="Advantage_system5.4", region="eu-central-1", token="DEV-c89566dc7b9419a5ddbc714bf36557ee3dcf3a4a")
        embedding_compose = EmbeddingComposite(sampler)
        sampleset = embedding_compose.sample(bqm, num_reads=10)
        sampleset.variables
        solver_times.append(time.perf_counter() - start_time_solver)

    print("Problem configuration times: ", problem_configuration_times)
    print("Solver times: ", solver_times)
    print("Average problem configuration time: ", sum(problem_configuration_times[1:])/(len(problem_configuration_times)-1))
    print("Average solver time: ", sum(solver_times[1:])/(len(solver_times)-1))



# sampler = dimod.SimulatedAnnealingSampler()
