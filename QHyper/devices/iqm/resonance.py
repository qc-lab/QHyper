from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit, transpile as qiskit_transpile
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends import (
    IQMFakeAdonis, IQMFakeAphrodite, IQMFakeApollo, IQMFakeDeneb,
)
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

from QHyper.devices.iqm.base import IQMDevice, IQM_BACKEND_URLS


IQM_FAKE_BACKENDS = {
    "adonis": IQMFakeAdonis,
    "apollo": IQMFakeApollo,
    "aphrodite": IQMFakeAphrodite,
    "deneb": IQMFakeDeneb,
    "garnet": IQMFakeGarnet,
}


@dataclass
class ResonanceDevice(IQMDevice):
    """
    Device running on an IQM Resonance QPU or on a local simulator.

    Attributes
    ----------
    type : str
        Type of the device the problem is solved on.
        Either 'simulator' or 'qpu' (IQM Resonance) is accepted.
    backend : str | None, default None
        For type 'qpu' name of the IQM machine: 'garnet', 'emerald' or
        'sirius' (required). For type 'simulator' name of the IQM fake
        backend: 'adonis', 'apollo', 'aphrodite', 'deneb' or 'garnet'
        (default 'garnet').
    token : str | None, default None
        IQM Resonance access token. Not used by the simulator.
    """

    type: str
    backend: str | None = None
    token: str | None = None

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type == 'simulator':
            if self.token is not None:
                raise ValueError(
                    "Simulator device does not use token/project/resource_name"
                )
            self.backend = str(self.backend or 'garnet').lower()
            if self.backend not in IQM_FAKE_BACKENDS:
                raise ValueError(
                    "IQM simulator device requires 'backend' set to one of: "
                    f"{sorted(IQM_FAKE_BACKENDS)}"
                )
        elif self.type == 'qpu':
            backend_key = (
                str(self.backend).lower() if self.backend is not None else ''
            )
            if backend_key not in IQM_BACKEND_URLS:
                raise ValueError(
                    "IQM qpu device requires 'backend' set to one of: "
                    f"{sorted(IQM_BACKEND_URLS)}"
                )
        else:
            raise ValueError(
                "'solver.device.type' must be either 'simulator' or 'qpu'"
            )

    def make_backend(self) -> Any:
        if self.type == 'simulator':
            return IQM_FAKE_BACKENDS[self.backend]()

        if self.token is not None:
            os.environ["IQM_TOKEN"] = self.token
        url = IQM_BACKEND_URLS[str(self.backend).lower()]
        return IQMProvider(url=url).get_backend()

    def transpile(self, qc: QuantumCircuit, backend: Any) -> QuantumCircuit:
        if self.type == 'simulator':
            return qiskit_transpile(qc, backend=backend)
        return qiskit_transpile(qc, backend=backend, optimization_level=3)
