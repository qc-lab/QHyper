import pennylane as qml
from pennylane import numpy as np
import timeit

qubits_num = 10
shots_num = 1000
num_of_parameterized_gates = 1000
ratio_imprim = 0.6

weights = np.random.uniform(low=0, high=2*np.pi, size=(num_of_parameterized_gates
        )).reshape(1, -1)

dev = qml.device("default.qubit", wires=qubits_num, shots=shots_num)


@qml.qnode(dev)
def circuit(weights, qubits_num, seed=42):
    qml.RandomLayers(weights=weights, wires=range(qubits_num), ratio_imprim=
        ratio_imprim, seed=seed)

    return qml.sample()

function_call = lambda: circuit(weights, qubits_num)
execution_time = timeit.timeit(stmt=function_call, number=10)
avg_execution_time_in_sec = execution_time / 10

print(f"Execution time: {avg_execution_time_in_sec} sec")

