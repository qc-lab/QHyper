from dataclasses import dataclass
import numpy as np
from typing import Optional, Any, cast
import numpy.typing as npt

from QHyper.problems.base import Problem

from QHyper.optimizers import OPTIMIZERS_BY_NAME
from QHyper.optimizers.base import Optimizer
from .pqc.base import PQC

from QHyper.solvers.base import Solver
from QHyper.solvers.vqa import PQC_BY_NAME


@dataclass
class OptWrapper:
    pqc: PQC
    problem: Problem
    hyper_params: npt.NDArray[np.float64]

    def __call__(self, args: npt.NDArray[np.float64]) -> float:
        return self.pqc.run_opt(self.problem, args, self.hyper_params)


@dataclass
class Wrapper:
    pqc: PQC
    problem: Problem
    optimizer: Optional[Optimizer]
    params_config: dict[str, Any]

    def __call__(self, hargs: npt.NDArray[np.float64]) -> float:
        opt_args = self.pqc.get_opt_args(self.params_config, hyper_args=hargs)
        opt_wrapper = OptWrapper(self.pqc, self.problem, hargs)

        if self.optimizer:
            value, _ = self.optimizer.minimize(opt_wrapper, opt_args)
        else:
            value = opt_wrapper(opt_args)
        return value


@dataclass
class VQA(Solver):
    problem: Problem
    pqc: PQC
    optimizer: Optional[Optimizer]

    def __init__(
            self,
            problem: Problem,
            pqc: PQC | str = "",
            optimizer: Optimizer | str = "",
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

    def solve(self, params_inits: dict[str, Any],
              hyper_optimizer: Optional[Optimizer] = None) -> Any:
        hyper_args = self.pqc.get_hopt_args(params_inits)

        if hyper_optimizer:
            # TODO
            wrapper = Wrapper(
                self.pqc, self.problem, self.optimizer, params_inits)
            _, best_hargs = hyper_optimizer.minimize(wrapper, hyper_args)
        else:
            best_hargs = hyper_args

        opt_args = self.pqc.get_opt_args(params_inits, hyper_args=best_hargs)
        opt_wrapper = OptWrapper(self.pqc, self.problem, hyper_args)
        if self.optimizer:
            _, best_opt_args = self.optimizer.minimize(opt_wrapper, opt_args)
        else:
            best_opt_args = opt_args

        return self.pqc.get_params_init_format(best_opt_args, best_hargs)

    def evaluate(self, params_inits: dict[str, Any]) -> float:
        hyper_args = self.pqc.get_hopt_args(params_inits)
        opt_args = self.pqc.get_opt_args(params_inits)
        return self.pqc.run_opt(self.problem, opt_args, hyper_args)
