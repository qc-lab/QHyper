from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any

from qiskit import QuantumCircuit

from QHyper.util import remap_keys
from QHyper.devices.base import Device


IQM_BACKEND_URLS = {
    "garnet": "https://cocos.resonance.meetiqm.com/garnet",
    "emerald": "https://cocos.resonance.meetiqm.com/emerald",
    "sirius": "https://cocos.resonance.meetiqm.com/sirius",
}


class IQMDevice(Device, ABC):
    """Abstract base class for IQM device configurations."""

    @abstractmethod
    def make_backend(self) -> Any:
        """
        Create the backend the circuits are run on.

        Returns
        -------
        Any
            Qiskit backend instance.
        """

        ...

    @abstractmethod
    def transpile(self, qc: QuantumCircuit, backend: Any) -> QuantumCircuit:
        """
        Transpile the circuit to the gate set of the given backend.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to transpile.
        backend : Any
            Backend created by make_backend.

        Returns
        -------
        QuantumCircuit
            Circuit that can be run on the backend.
        """

        ...

    @classmethod
    def from_config(cls, config: Any) -> "IQMDevice":
        from QHyper.devices.iqm.resonance import ResonanceDevice
        from QHyper.devices.iqm.vlq import VLQDevice

        if isinstance(config, IQMDevice):
            return config
        if not isinstance(config, dict):
            raise ValueError("'solver.device' must be a mapping")

        cfg = remap_keys(
            config,
            ['type', 'name', 'backend', 'project', 'resource_name', 'token'],
        )
        device_type = str(cfg.get('type', '')).lower()
        device_name = str(cfg.get('name', '')).lower()

        if device_type == 'simulator':
            if device_name in ('it4i', 'lexis', 'vlq'):
                return _build(VLQDevice, cfg)
            return _build(ResonanceDevice, cfg)
        if device_type == 'qpu':
            if device_name in ('iqm.resonance', 'iqm', 'resonance'):
                return _build(ResonanceDevice, cfg)
            if device_name in ('it4i', 'lexis', 'vlq'):
                return _build(VLQDevice, cfg)
            if not device_name:
                raise ValueError(
                    "'solver.device.name' must be a non-empty string"
                )
            raise ValueError(
                f"Unsupported IQM qpu device name '{device_name}'. "
                "Use 'iqm.resonance' for IQM or 'it4i'/'lexis' for LEXIS"
            )
        raise ValueError(
            "'solver.device.type' must be either 'simulator' or 'qpu'"
        )


def _build(target: type, cfg: dict[str, Any]) -> "IQMDevice":
    names = {f.name for f in fields(target)}
    unknown = [k for k in cfg if k not in names and k != 'name']
    if unknown:
        raise ValueError(
            f"{target.__name__} does not accept config keys: {sorted(unknown)}"
        )
    return target(**{k: v for k, v in cfg.items() if k in names})
