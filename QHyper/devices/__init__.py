"""
This module contains device configurations, classes that describe where a
solver is run and know how to create the backend, sampler or simulator used
by the given platform.
Devices are kept separate from solvers, so the same solver can be run on a
local simulator or on real hardware just by changing the device.

.. code-block:: python

    from QHyper.devices.pennylane import PennyLaneDevice

.. rubric:: Interface

.. autosummary::
    :toctree: generated

    Device -- Base class for devices.

.. rubric:: Available devices

.. autosummary::
    :toctree: generated

    pennylane.PennyLaneDevice -- PennyLane device configuration.
    iqm.ResonanceDevice -- IQM Resonance device configuration.
    iqm.VLQDevice -- VLQ device configuration.
    dwave.DWaveDevice -- D-Wave device configuration.

"""

from QHyper.devices.base import Device
