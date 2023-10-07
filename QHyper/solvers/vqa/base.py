from dataclasses import dataclass
import numpy as np
from typing import Any, cast
import numpy.typing as npt

from QHyper.problems.base import Problem

from QHyper.optimizers import (
    OPTIMIZERS_BY_NAME, Dummy, Optimizer, OptimizationResult)
from .pqc.base import PQC

from QHyper.solvers.base import Solver, SolverResult
from QHyper.solvers.vqa import PQC_BY_NAME


@dataclass
class LocalOptimizerFunction:
    pqc: PQC
    problem: Problem
    hyper_params: npt.NDArray[np.float64]

    def __call__(self, args: npt.NDArray[np.float64]) -> OptimizationResult:
        return self.pqc.run_opt(self.problem, args, self.hyper_params) 


@dataclass
class GlobalOptimizerFunction:
    pqc: PQC
    problem: Problem
    optimizer: Optimizer
    params_config: dict[str, Any]

    def __call__(self, hargs: npt.NDArray[np.float64]) -> float:
        opt_args = self.pqc.get_opt_args(self.params_config, hyper_args=hargs)
        opt_wrapper = LocalOptimizerFunction(self.pqc, self.problem, hargs)
        return self.optimizer.minimize(opt_wrapper, opt_args)


@dataclass
class VQA(Solver):
    problem: Problem
    pqc: PQC
    optimizer: Optimizer = Dummy()
    hyper_optimizer: Optimizer | None = None

    def __init__(
            self,
            problem: Problem,
            pqc: PQC | str = "",
            optimizer: Optimizer | str = "",
            hyper_optimizer: Optimizer | None = None,
            config: dict[str, dict[str, Any]] = {}
    ) -> None:
        self.problem = problem
        self.hyper_optimizer = hyper_optimizer

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
                self.optimizer = Dummy()
            else:
                if optimizer == "":
                    optimizer = config['optimizer'].pop('type')
                self.optimizer = OPTIMIZERS_BY_NAME[cast(str, optimizer)](
                    **config.get('optimizer', {}))
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer

    def solve(self, params_inits: dict[str, Any]) -> SolverResult:
        hyper_args = self.pqc.get_hopt_args(params_inits)

        if self.hyper_optimizer:
            wrapper = GlobalOptimizerFunction(
                self.pqc, self.problem, self.optimizer, params_inits)
            res = self.hyper_optimizer.minimize(wrapper, hyper_args)
            best_hargs = res.params
        else:
            best_hargs = hyper_args
        
        opt_args = self.pqc.get_opt_args(params_inits, hyper_args=best_hargs)
        opt_wrapper = LocalOptimizerFunction(
                self.pqc, self.problem, best_hargs)
        opt_res = self.optimizer.minimize(opt_wrapper, opt_args)

        return SolverResult(
            self.pqc.run_with_probs(self.problem, opt_res.params, best_hargs),
            self.pqc.get_params_init_format(opt_res.params, best_hargs)
        )
