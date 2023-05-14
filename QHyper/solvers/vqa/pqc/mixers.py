import pennylane as qml

from typing import Iterable, Callable


def mixer(wires: Iterable[str]) -> qml.Hamiltonian:
    return qml.qaoa.x_mixer(wires)


MIXERS_BY_NAME: dict[str, Callable[[Iterable[str]], qml.Hamiltonian]] = {
    'pl_x_mixer': mixer
}
