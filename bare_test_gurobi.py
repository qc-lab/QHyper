import gurobipy as gp
import networkx as nx
import time

edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 9), (0, 14), (1, 2), (1, 3), (1, 5), (1, 9), (1, 10), (1, 11), (1, 12), (1, 14), (2, 5), (2, 9), (2, 12), (2, 14), (3, 4), (3, 7), (3, 8), (3, 9), (3, 10), (4, 6), (4, 7), (4, 10), (4, 12), (4, 13), (5, 9), (5, 10), (6, 8), (6, 12), (7, 9), (7, 10), (8, 10), (8, 14), (9, 10), (9, 11), (11, 12), (11, 13), (12, 14)]
graph = nx.Graph(edges)

if __name__ == '__main__':
    
    problem_configuration_times = []
    solver_times =[]
    
    for i in range(10):
        start_time_problem = time.time()
        model = gp.Model()
        x = model.addVars(range(len(graph.nodes)), vtype=gp.GRB.BINARY) 
        cost_function = 0
        for i,j in graph.edges:
            cost_function += -x[i] -x[j] + 2*x[i]*x[j]
        model.setObjective(cost_function, gp.GRB.MINIMIZE)
        problem_configuration_times.append(time.time() - start_time_problem)
        
        start_time_solver = time.time()
        model.optimize()
        solver_times.append(time.time() - start_time_solver)
    
    print("Problem configuration times: ", problem_configuration_times)
    print("Solver times: ", solver_times)
    print("Average problem configuration time: ", sum(problem_configuration_times[1:])/(len(problem_configuration_times)-1))
    print("Average solver time: ", sum(solver_times[1:])/(len(solver_times)-1))
    