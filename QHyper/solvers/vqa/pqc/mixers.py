# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import pennylane as qml

from typing import Iterable, Callable


def mixer(wires: Iterable[str]) -> qml.Hamiltonian:
    return qml.qaoa.x_mixer(wires)


MIXERS_BY_NAME: dict[str, Callable[[Iterable[str]], qml.Hamiltonian]] = {
    'pl_x_mixer': mixer
}
