from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dwave.cloud import Client
from dwave.cloud.regions import get_regions
from dwave.system import (
    DWaveSampler,
    LeapHybridCQMSampler,
    LeapHybridDQMSampler,
)

from QHyper.util import normalize_key, remap_keys
from QHyper.devices.base import Device


DEVICE_NAME = 'DWaveSampler'

SAMPLERS = {
    'advantage': (DWaveSampler,
                  lambda s: s.qpu),
    'cqm': (LeapHybridCQMSampler,
            lambda s: 'cqm' in s.supported_problem_types),
    'dqm': (LeapHybridDQMSampler,
            lambda s: 'dqm' in s.supported_problem_types),
}


@dataclass
class DWaveDevice(Device):
    """
    Device configuration for the D-Wave solvers.

    Attributes
    ----------
    type : str, default 'qpu'
        Type of the device the problem is solved on.
        Only 'qpu' is accepted, every D-Wave solver (Advantage, CQM, DQM)
        runs on a remote machine.
    name : str, default 'DWaveSampler'
        Name of the device. Only 'DWaveSampler' is accepted.
    backend : str | None, default None
        Name of the backend, e.g. 'Advantage_system5.4'. If not provided,
        the default backend available for the token is used.
    region : str | None, default None
        Region the backend is looked up in, e.g. 'eu-central-1'. If not
        provided, the default region is used.
    token : str | None, default None
        D-Wave Leap API token. If not provided, the DWAVE_API_TOKEN
        environment variable is used.
    """

    type: str = 'qpu'
    name: str = DEVICE_NAME
    backend: str | None = None
    region: str | None = None
    token: str | None = None

    def __post_init__(self) -> None:
        self.type = str(self.type).lower()
        if self.type != 'qpu':
            raise ValueError(
                "'device.type' must be 'qpu', every D-Wave solver "
                "(advantage, cqm, dqm) runs on a remote machine"
            )

        if normalize_key(self.name) != normalize_key(DEVICE_NAME):
            raise ValueError(
                f"'device.name' must be '{DEVICE_NAME}', got '{self.name}'"
            )
        self.name = DEVICE_NAME

        if self.region is not None:
            regions = [region.code for region in get_regions()]
            if self.region not in regions:
                raise ValueError(
                    f"Unknown D-Wave region '{self.region}'. "
                    f"Available regions: {sorted(regions)}"
                )

    @classmethod
    def from_config(cls, config: Any) -> "DWaveDevice":
        if isinstance(config, cls):
            return config
        if not isinstance(config, dict):
            raise ValueError("'device' must be a mapping")
        return cls(**remap_keys(
            config, ['type', 'name', 'backend', 'region', 'token']))

    def make_sampler(self, solver_name: str) -> Any:
        if solver_name not in SAMPLERS:
            raise ValueError(
                f"Unknown solver '{solver_name}', "
                f"expected one of {sorted(SAMPLERS)}"
            )
        sampler, runs_on = SAMPLERS[solver_name]

        kwargs: dict[str, Any] = {}
        token = self.token or os.environ.get('DWAVE_API_TOKEN')
        if token is not None:
            kwargs['token'] = token
        if self.region is not None:
            kwargs['region'] = self.region

        with Client.from_config(**kwargs) as client:
            machines = sorted(
                s.name for s in client.get_solvers() if runs_on(s))

        where = (f"region '{self.region}'" if self.region
                 else 'the default region')
        if not machines:
            raise ValueError(
                f"No machine for the '{solver_name}' solver is available "
                f"in {where} with this token"
            )
        if self.backend is not None and self.backend not in machines:
            raise ValueError(
                f"'{self.backend}' is not a valid machine for the "
                f"'{solver_name}' solver in {where}. "
                f"Available machines: {machines}"
            )

        if self.backend is not None:
            kwargs['solver'] = self.backend
        return sampler(**kwargs)
