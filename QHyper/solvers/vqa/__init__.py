from typing import Type

# === PQC ===

from .pqc.base import PQC
from .pqc.h_qaoa import HQAOA
from .pqc.qaoa import QAOA

PQC_BY_NAME: dict[str, Type[PQC]] = {
    'hqaoa': HQAOA,
    'qaoa': QAOA,
}

# === Evaluation functions ===

from .eval_funcs.base import EvalFunc
from .eval_funcs.expval import ExpVal
from .eval_funcs.wfeval import WFEval

EVAL_FUNCS_BY_NAME: dict[str, Type[EvalFunc]] = {
    'expval': ExpVal,
    'wfeval': WFEval,
}
