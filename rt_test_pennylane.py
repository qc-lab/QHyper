from QHyper.solvers import solver_from_config
import time

solver_config = {
    "solver": {
        "name": "QAOA",
        "category": "gate_based",
        "platform": "pennylane",
        "layers": 2,
        "gamma": {
            "init": [0.5]*2,
        },
        "beta": {
            "init": [1]*2,
        },
        "penalty_weights": [1],
        "optimizer": {
            "type": "qml",
            "name": "adam",
            "steps": 70,
        },
    },
    "problem": {
        "type": "maxcut",
        "edges": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (2, 1), (3, 5)]  #, (1, 2), (1, 3), (1, 5), (1, 9), (1, 10), (1, 11), (1, 12), (1, 14), (2, 5), (2, 9), (2, 12), (2, 14), (3, 4), (3, 7), (3, 8), (3, 9), (3, 10), (4, 6), (4, 7), (4, 10), (4, 12), (4, 13), (5, 9), (5, 10), (6, 8), (6, 12), (7, 9), (7, 10), (8, 10), (8, 14), (9, 10), (9, 11), (11, 12), (11, 13), (12, 14)],
        # "edges": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 9), (0, 14), (1, 2), (1, 3), (1, 5), (1, 9), (1, 10), (1, 11), (1, 12), (1, 14), (2, 5), (2, 9), (2, 12), (2, 14), (3, 4), (3, 7), (3, 8), (3, 9), (3, 10), (4, 6), (4, 7), (4, 10), (4, 12), (4, 13), (5, 9), (5, 10), (6, 8), (6, 12), (7, 9), (7, 10), (8, 10), (8, 14), (9, 10), (9, 11), (11, 12), (11, 13), (12, 14)]
            # [(0, 2), (0, 11), (0, 13), (0, 16), (1, 3), (1, 4), (1, 8), (1, 11), (1, 15), (2, 4), (3, 4), (3, 8), (3, 16), (4, 7), (4, 10), (4, 15), (4, 19), (5, 6), (5, 10), (5, 12), (6, 11), (6, 13), (6, 15), (7, 14), (7, 18), (8, 9), (8, 15), (8, 17), (8, 18), (9, 11), (9, 14), (9, 16), (12, 18), (13, 14), (14, 15), (18, 19)],
    }
}

if __name__ == "__main__":

    problem_configuration_times = []
    solver_times =[]

    for i in range(10):
        start_time_problem = time.perf_counter()
        vqa = solver_from_config(solver_config)
        problem_configuration_times.append(time.perf_counter() - start_time_problem)

        start_time_solver = time.perf_counter()
        res = vqa.solve()
        solver_times.append(time.perf_counter() - start_time_solver)
        print("Time: ", time.perf_counter() - start_time_solver)

    print("Problem configuration times: ", problem_configuration_times)
    print("Solver times: ", solver_times)
    print("Average problem configuration time: ", sum(problem_configuration_times[1:])/(len(problem_configuration_times)-1))
    print("Average solver time: ", sum(solver_times[1:])/(len(solver_times)-1))
