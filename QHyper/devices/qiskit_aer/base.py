# This project has received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) 
# under grant agreement No 101159808 (EuroQHPC-Integration).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile as qiskit_transpile
from qiskit_aer import AerSimulator

from QHyper.util import remap_keys
from QHyper.devices.base import Device


@dataclass
class AerDevice(Device):
    """
    Device configuration for the local Qiskit Aer simulator.

    Attributes
    ----------
    type : str, default 'simulator'
        Type of the device the problem is solved on.
        For now only 'simulator' is accepted.
    """

    type: str = 'simulator'

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type != 'simulator':
            raise ValueError(
                "'device.type' must be 'simulator', Qiskit Aer is a local "
                "simulator"
            )

    @classmethod
    def from_config(cls, config: Any) -> "AerDevice":
        if isinstance(config, cls):
            return config
        if not isinstance(config, dict):
            raise ValueError("'device' must be a mapping")
        return cls(**remap_keys(config, ['type']))

    def make_backend(self) -> AerSimulator:
        return AerSimulator()

    def transpile(self, qc: QuantumCircuit, backend: Any) -> QuantumCircuit:
        return qiskit_transpile(qc, backend=backend)
