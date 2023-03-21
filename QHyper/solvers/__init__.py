# from .cqm import cqm
from .vqa.base import VQA

SOLVERS = {
    # 'cqm': cqm.CQM,
    # 'pl_qaoa': qaoa.QAOA,
    # 'pl_wf_qaoa': wf_qaoa.WFQAOA,
    # 'pl_h_qaoa': h_qaoa.HQAOA,
    # 'qaoa': VQA(optimizer='scipy', pqc="qiskit_qaoa", eval_func='qaoa'),
    # 'wfqaoa': VQA(optimizer='scipy', pqc="qaoa", eval_func='wfeval')
    # 'vqa': CQM,
    # 'qurobi': QUROBI
}


# solver = SOLVERS['vqa'](optimizer='scipy', pqc="hqaoa")
# solver.solve()
