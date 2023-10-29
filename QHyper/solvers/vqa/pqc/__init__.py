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
