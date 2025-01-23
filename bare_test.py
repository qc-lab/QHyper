import time
import networkx as nx
from collections import defaultdict

import gurobipy as gp
import pennylane as qml
import dimod
import numpy as np
from QHyper.solvers import solver_from_config
from dwave.system import DWaveSampler, EmbeddingComposite


DWAVE_TOKEN = "DWAVE-TOKEN"

QAOA_STEP_SIZE = 0.005
QAOA_STEPS = 70
QAOA_LAYERS = 2
QAOA_GAMMA = [0.5]*QAOA_LAYERS
QAOA_BETA = [1]*QAOA_LAYERS


def get_small_graphs():
    sizes = [(5, 5), (10, 25), (10, 30), (12, 75), (15, 50), (15, 150)]

    graphs = []
    for n, e in sizes:
        graphs.append(nx.gnm_random_graph(n, e, seed=123))
    return graphs


def get_graphs():
    sizes = [(5, 5), (10, 25), (15, 50), (20, 100),
             (25, 200), (30, 400), (35, 700)]

    graphs = []
    for n, e in sizes:
        graphs.append(nx.gnm_random_graph(n, e, seed=123))

    return graphs


def bare_pennylane(graph):
    start_time_problem = time.perf_counter()
    cost_h, mixer_h = qml.qaoa.maxcut(graph)

    params = qml.numpy.array([QAOA_GAMMA, QAOA_BETA], requires_grad=True)

    dev = qml.device("default.qubit", wires=range(len(graph.nodes)))

    def qaoa_layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(beta, mixer_h)

    def circuit(params):
        for n in graph.nodes:
            qml.Hadamard(wires=n)
        qml.layer(qaoa_layer, QAOA_LAYERS, params[0], params[1])

    @qml.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h.simplify())

    optimizer = qml.AdamOptimizer(stepsize=QAOA_STEP_SIZE)
    for i in range(QAOA_STEPS):
        params = optimizer.step(cost_function, params)

    @qml.qnode(dev)
    def probability_circuit(gamma, beta):
        circuit([gamma, beta])
        return qml.probs(wires=graph.nodes)

    probability_circuit(params[0], params[1])

    return time.perf_counter() - start_time_problem


def bare_gurobi(graph):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    start_time_problem = time.perf_counter()
    model = gp.Model(env=env)
    model.setParam("Threads", 1)
    x = model.addVars(range(len(graph.nodes)), vtype=gp.GRB.BINARY)
    cost_function = 0
    for i, j in graph.edges:
        cost_function += -x[i] - x[j] + 2*x[i]*x[j]
    model.setObjective(cost_function, gp.GRB.MINIMIZE)

    model.optimize()
    allvars = model.getVars()

    return time.perf_counter() - start_time_problem


def bare_advantage(graph):
    start_time_problem = time.perf_counter()
    linear = defaultdict(int)
    quadratic = defaultdict(int)

    for i, j in graph.edges:
        linear[i] += -1
        linear[j] += -1
        quadratic[(i, j)] += 2

    bqm = dimod.BinaryQuadraticModel(
        linear, quadratic, 0.0, vartype=dimod.BINARY)

    sampler = DWaveSampler(solver="Advantage_system5.4", region="eu-central-1",
                           token=DWAVE_TOKEN)
    embedding_compose = EmbeddingComposite(sampler)
    sampleset = embedding_compose.sample(bqm, num_reads=10)

    # accessing any attribute of sampleset takes quite some time
    sampleset.variables

    return time.perf_counter() - start_time_problem


def qhyper_qaoa(graph):
    solver_config = {
        "solver": {
            "name": "QAOA",
            "category": "gate_based",
            "platform": "pennylane",
            "layers": QAOA_LAYERS,
            "gamma": {
                "init": QAOA_GAMMA,
            },
            "beta": {
                "init": QAOA_BETA,
            },
            "optimizer": {
                "type": "qml",
                "name": "adam",
                "steps": QAOA_STEPS,
                "stepsize": QAOA_STEP_SIZE,
            },
        },
        "problem": {
            "type": "maxcut",
            "edges": graph.edges
        }
    }
    start_time_problem = time.perf_counter()
    vqa = solver_from_config(solver_config)

    vqa.solve()

    return time.perf_counter() - start_time_problem


def qhyper_gurobi(graph):
    solver_config = {
        "solver": {
            "name": "gurobi",
        },
        "problem": {
            "type": "maxcut",
            "edges": graph.edges
        }
    }
    start_time_problem = time.perf_counter()
    vqa = solver_from_config(solver_config)

    vqa.solve()

    return time.perf_counter() - start_time_problem


def qhyper_advantage(graph):
    solver_config = {
        "solver": {
            "name": "Advantage",
            "category": "quantum_annealing",
            "platform": "dwave",
            "num_reads": 10,
            "token": DWAVE_TOKEN
        },
        "problem": {
            "type": "maxcut",
            "edges": graph.edges
        }
    }

    start_time_problem = time.perf_counter()
    vqa = solver_from_config(solver_config)

    vqa.solve()

    return time.perf_counter() - start_time_problem


def run_test(graph, func, iter, silent=False):
    solver_times = []
    for i in range(iter):
        s_time = func(graph)
        solver_times.append(s_time)
    if not silent:
        print(f"{sum(solver_times)/iter:.3g} ± {np.std(solver_times):.3g}")
    return solver_times


def run_qaoa_tests(num_of_reps):
    print("Gate-base test")
    for g in get_small_graphs():
        print(f"{len(g.nodes)} nodes, {len(g.edges)} edges")
        run_test(g, bare_pennylane, num_of_reps)
        run_test(g, qhyper_qaoa, num_of_reps)


def run_gurobi_tests(num_of_reps):
    print("Gurobi test")
    for g in get_graphs():
        print(f"{len(g.nodes)} nodes, {len(g.edges)} edges")
        run_test(g, bare_gurobi, num_of_reps)
        run_test(g, qhyper_gurobi, num_of_reps)


def run_advantage_tests(num_of_reps):
    print("Advantage test")
    # First advantage test is really slow, so we run it separately before tests
    run_test(nx.random_geometric_graph(3, .1), bare_advantage, 1, True)

    for g in get_graphs():
        print(f"{len(g.nodes)} nodes, {len(g.edges)} edges")
        run_test(g, bare_advantage, num_of_reps)
        run_test(g, qhyper_advantage, num_of_reps)


if __name__ == '__main__':
    run_qaoa_tests(10)
    run_gurobi_tests(10)
    run_advantage_tests(10)
