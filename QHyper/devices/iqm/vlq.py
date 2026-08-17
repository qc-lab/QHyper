from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile as qiskit_transpile
from py4lexis.session import LexisSession
from qaas.client import QProvider
from qaas.client.backend import transpile as lexis_transpile

from QHyper.devices.iqm.base import IQMDevice
from QHyper.devices.iqm.fake_vlq import FakeVLQ


@dataclass
class VLQDevice(IQMDevice):
    """
    Device running on VLQ, the IQM machine accessed through 
    the LEXIS platform.

    Attributes
    ----------
    type : str
        Type of the device the problem is solved on.
        Either 'simulator' or 'qpu' (VLQ) is accepted.
    project : str | None, default None
        Name of the LEXIS project. Required for type 'qpu'.
    resource_name : str | None, default None
        Name of the LEXIS resource. Required for type 'qpu'.
    token : str | None, default None
        LEXIS access token. If not provided, a LexisSession
        login is used to obtain it. Not used by the simulator.
    """

    type: str
    project: str | None = None
    resource_name: str | None = None
    token: str | None = None

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type == 'qpu':
            if not self.project or not self.resource_name:
                raise ValueError(
                    "LEXIS device requires 'project' and 'resource_name'"
                )
        elif self.type == 'simulator':
            if (self.project is not None or self.resource_name is not None
                    or self.token is not None):
                raise ValueError(
                    "Simulator (fake VLQ) device does not use "
                    "project/resource_name/token"
                )
        else:
            raise ValueError(
                "'solver.device.type' must be either 'simulator' or 'qpu'"
            )

    def make_backend(self) -> Any:
        if self.type == 'simulator':
            return FakeVLQ()

        token = self.token
        if token is None:
            token = LexisSession().get_access_token()
        return QProvider(self.project, token).get_backend(self.resource_name)

    def transpile(self, qc: QuantumCircuit, backend: Any) -> QuantumCircuit:
        if self.type == 'simulator':
            return qiskit_transpile(qc, backend=backend)
        return lexis_transpile(qc, backend, optimize_single_qubits=False)
