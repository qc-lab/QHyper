# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


"""
=============================================================
Parametrized Quantum Circuits (:mod:`QHyper.solvers.vqa.pqc`)
=============================================================

.. currentmodule:: QHyper.solvers.vqa.pqc

Interfaces
==========

.. autosummary::
    :toctree: generated/

    PQC  -- Base class for parametrized quantum circuits.

Implementations
===============

.. autosummary::
    :toctree: generated/

    QAOA -- Quantum Approximate Optimization Algorithm.
    HQAOA -- `<https://www.iccs-meeting.org/archive/iccs2023/papers/140770117.pdf>`
    WFQAOA -- `<https://www.iccs-meeting.org/archive/iccs2023/papers/140770117.pdf>`
    QML_QAOA -- QAOA with specific interface to enable some more complex PennyLane optimizators.

"""


from typing import Type

from .base import PQC
from .h_qaoa import HQAOA
from .qaoa import QAOA
from .wf_qaoa import WFQAOA
from .qml_qaoa import QML_QAOA

PQC_BY_NAME: dict[str, Type[PQC]] = {
    'hqaoa': HQAOA,
    'qaoa': QAOA,
    'wfqaoa': WFQAOA,
    'qml_qaoa': QML_QAOA,
}
