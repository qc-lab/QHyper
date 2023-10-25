import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="autograd")
def circuit(params):
    # |psi_0>: state preparation
    qml.Hadamard(wires=0)
   # qml.RY(np.pi/2, wires=0)

    # V0(theta0, theta1): Parametrized layer 0
    qml.RZ(params[0],wires=0)
    qml.RX(params[1],wires=0)
 #   qml.RZ(params[2],wires=0)
 #   qml.RX(params[3],wires=0)
   
   
    return qml.expval(qml.PauliZ(0))


params = np.array([0.1e-13, 0.1e-13])
print(qml.metric_tensor(circuit, approx='diag')(params))