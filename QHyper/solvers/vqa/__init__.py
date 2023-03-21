# === PQC ===

from .pqc.base import PQC
from .pqc.h_qaoa import HQAOA
from .pqc.qaoa import QAOA

PQC_BY_NAME: dict[str, PQC] = {
    'hqaoa': HQAOA,
    'qaoa': QAOA,
}

# === Evaluation functions ===

from .eval_funcs.base import EvalFunc
from .eval_funcs.expval import ExpVal
from .eval_funcs.wfeval import WFEval

EVAL_FUNCS_BY_NAME: dict[str, EvalFunc] = {
    'expval': ExpVal,
    'wfeval': WFEval,
}
