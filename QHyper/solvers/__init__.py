from typing import Type

from .base import Solver

from .vqa.base import VQA
from .gurobi.gurobi import Gurobi
from .cqm.cqm import CQM


SOLVERS: dict[str, Type[Solver]] = {
    # 'cqm': cqm.CQM,
    # 'pl_qaoa': qaoa.QAOA,
    # 'pl_wf_qaoa': wf_qaoa.WFQAOA,
    # 'pl_h_qaoa': h_qaoa.HQAOA,
    # 'qaoa': VQA(optimizer='scipy', pqc="qiskit_qaoa", eval_func='qaoa'),
    # 'wfqaoa': VQA(optimizer='scipy', pqc="qaoa", eval_func='wfeval')
    'vqa': VQA,
    'gurobi': Gurobi,
    'cqm': CQM,
}


# solver = SOLVERS['vqa'](optimizer='scipy', pqc="hqaoa")
# solver.solve()
