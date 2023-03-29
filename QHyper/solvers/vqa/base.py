from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Any, cast
import numpy.typing as npt

from QHyper.problems.base import Problem

from QHyper.optimizers import OPTIMIZERS_BY_NAME
from QHyper.optimizers.base import Optimizer
from .pqc.base import PQC
from .eval_funcs.base import EvalFunc

from QHyper.solvers.base import Solver, SolverResults
from QHyper.solvers.vqa import EVAL_FUNCS_BY_NAME, PQC_BY_NAME


@dataclass
class OptWrapper:
    pqc: PQC
    eval_func: EvalFunc
    problem: Problem
    hyper_params: npt.NDArray[np.float64]
    # params_config: dict[str, Any]

    def __call__(self, args: npt.NDArray[np.float64]) -> float:
        results, weights = self.pqc.run(self.problem, args, self.hyper_params)
        # weights = self.params_config  # TODO
        return self.eval_func.evaluate(results, self.problem, weights)
    

@dataclass
class Wrapper:
    pqc: PQC
    eval_func: EvalFunc
    problem: Problem
    optimizer: Optional[Optimizer]
    params_config: dict[str, Any]

    # def run(self, args: list[float]) -> float:
    #     results = self.pqc.run(self.problem, args, self.params_config)
    #     weights = self.params_config  # TODO
    #     return self.eval_func.evaluate(results, self.problem, weights)

    def __call__(self, hargs: npt.NDArray[np.float64]) -> float:
        hyper_args, opt_args = self.pqc.get_params(self.params_config, hargs)
        opt_wrapper = OptWrapper(self.pqc, self.eval_func, self.problem, hyper_args)
        # opt_args = self.pqc.get_opt(args)
        if self.optimizer:
            value, _ = self.optimizer.minimize(opt_wrapper, np.array(opt_args))
        else:
            value = opt_wrapper(opt_args)
        return value


@dataclass
class VQA(Solver):
    problem: Problem
    pqc: PQC
    optimizer: Optional[Optimizer]
    eval_func: EvalFunc

    def __init__(
            self,
            problem: Problem,
            pqc: PQC | str = "",
            optimizer: Optimizer | str = "",
            eval_func: EvalFunc | str = "",
            config: dict[str, dict[str, Any]] = {}
    ) -> None:
        self.problem = problem

        if isinstance(pqc, str):
            if pqc == "":
                try:
                    pqc = config['pqc'].pop('type')
                except KeyError:
                    raise Exception("Configuration for PQC was not provided")
            self.pqc = PQC_BY_NAME[cast(str, pqc)](**config.get('pqc', {}))
        else:
            self.pqc = pqc

        if isinstance(optimizer, str):
            if optimizer == "" and 'optimizer' not in config:
                self.optimizer = None
            else:
                if optimizer == "":
                    optimizer = config['optimizer'].pop('type')
                self.optimizer = OPTIMIZERS_BY_NAME[cast(str, optimizer)](
                    **config.get('optimizer', {}))
        else:
            self.optimizer = optimizer

        if isinstance(eval_func, str):
            if eval_func == "":
                try:
                    eval_func = config['eval_func'].pop('type')
                except KeyError:
                    raise Exception("Configuration for Eval Func was not provided")
            self.eval_func = EVAL_FUNCS_BY_NAME[cast(str, eval_func)](**config.get('eval_func', {}))
        else:
            self.eval_func = eval_func

    def solve(self, params_inits: dict[str, Any], hyper_optimizer: Optional[Optimizer] = None) -> Any:
        hyper_args, _ = self.pqc.get_params(params_inits)
        wrapper = Wrapper(self.pqc, self.eval_func, self.problem, self.optimizer, params_inits)

        if hyper_optimizer:
            _, best_args = hyper_optimizer.minimize(wrapper, hyper_args)
        else:
            best_args = hyper_args

        hyper_args, opt_args = self.pqc.get_params(params_inits, best_args)
        opt_wrapper = OptWrapper(self.pqc, self.eval_func, self.problem, hyper_args)
        if self.optimizer:
            value, best_opt_args = self.optimizer.minimize(opt_wrapper, opt_args)
        else:
            value, best_opt_args = opt_wrapper(opt_args), opt_args

        return value, best_opt_args, hyper_args
