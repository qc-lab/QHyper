from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pennylane as qml

from QHyper.util import remap_keys
from QHyper.devices.base import Device


SIMULATOR_DEVICES = {
    'default.qubit', 'default.mixed', 'default.tensor',
    'lightning.qubit', 'lightning.gpu', 'lightning.kokkos', 'lightning.tensor',
    'qiskit.aer', 'qiskit.basicsim',
}
QISKIT_REMOTE = 'qiskit.remote'


@dataclass
class PennyLaneDevice(Device):
    """
    Device configuration for the PennyLane based solvers.

    Attributes
    ----------
    type : str
        Type of the device the problem is solved on.
        Either 'simulator' or 'qpu' is accepted.
    name : str
        Name of the PennyLane device, e.g. 'default.qubit',
        or another plugin for QPUs.
    backend : str | None, default None
        Name of the backend, e.g. 'melbourne'. Required for
        'qiskit.remote', not used by the simulators.
    """

    type: str
    name: str
    backend: str | None = None

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("'device.name' must be a non-empty string")
        if self.backend is not None:
            self.backend = str(self.backend)

        name_key = self.name.lower()
        if self.type == 'simulator':
            if name_key == QISKIT_REMOTE:
                raise ValueError(
                    "'qiskit.remote' is a remote QPU device, not a simulator. "
                    "Use 'qiskit.aer' for local Aer simulation, or device.type "
                    "'qpu' with a hardware backend"
                )
        elif self.type == 'qpu':
            if name_key == QISKIT_REMOTE and not self.backend:
                raise ValueError(
                    "'qiskit.remote' requires 'device.backend' "
                    "(e.g. a hardware backend like 'melbourne')"
                )
            if name_key in SIMULATOR_DEVICES:
                raise ValueError(
                    f"'{self.name}' is a simulator, use device.type 'simulator' "
                    "or 'qiskit.remote' with a backend for a real qpu"
                )
        else:
            raise ValueError(
                "'device.type' must be either 'simulator' or 'qpu'"
            )

    @classmethod
    def from_config(cls, config: Any) -> "PennyLaneDevice":
        if isinstance(config, cls):
            return config
        if not isinstance(config, dict):
            raise ValueError("'device' must be a mapping")
        return cls(**remap_keys(config, ['type', 'name', 'backend']))

    def make_device(self, wires: Any) -> "qml.devices.LegacyDevice":
        kwargs = {'backend': self.backend} if self.backend else {}
        return qml.device(self.name, wires=wires, **kwargs)
