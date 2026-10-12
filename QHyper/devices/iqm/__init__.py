# This project has received funding from the European High Performance Computing Joint Undertaking (EuroHPC JU) 
# under grant agreement No 101159808 (EuroQHPC-Integration).

from QHyper.devices.iqm.base import IQMDevice
from QHyper.devices.iqm.resonance import ResonanceDevice, IQM_FAKE_BACKENDS, IQM_BACKEND_URL, IQMBackend


def __getattr__(name):
    # VLQDevice needs the optional 'vlq' extra (py4lexis, qaas)
    if name == "VLQDevice":
        from QHyper.devices.iqm.vlq import VLQDevice
        return VLQDevice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")