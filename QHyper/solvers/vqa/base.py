from dataclasses import dataclass
import numpy as np
from typing import Any, Optional
import numpy.typing as npt

from QHyper.problems.base import Problem

from QHyper.optimizers import (
    OPTIMIZERS_BY_NAME, Dummy, Optimizer, OptimizationResult)
from .pqc.base import PQC

from QHyper.solvers.base import Solver, SolverResult, SolverConfigException
from QHyper.solvers.vqa.pqc import PQC_BY_NAME


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
    params_inits: Optional[dict[str, Any]] = None

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'VQA':
        try:
            error_msg = "PQC configuration was not provided"
            pqc_config = config.pop('pqc')
            error_msg = "PQC type was not provided"
            pqc_type = pqc_config.pop('type')
            error_msg = f"There is no {pqc_type} PQC type"
            pqc = PQC_BY_NAME[pqc_type](**pqc_config)
        except KeyError:
            raise SolverConfigException(error_msg)

        if not (optimizer_config := config.pop('optimizer', None)):
            optimizer = Dummy()
        elif not (optimizer_type := optimizer_config.pop('type', None)):
            raise SolverConfigException("Optimizer type was not provided")
        elif not (optimizer_class := OPTIMIZERS_BY_NAME.get(
                optimizer_type, None)):
            raise SolverConfigException(
                f"There is no {optimizer_type} optimizer type")
        else:
            optimizer = optimizer_class(**optimizer_config)

        if not (hyper_optimizer_config := config.pop('hyper_optimizer', None)):
            hyper_optimizer = None
        elif not (hyper_optimizer_type := hyper_optimizer_config.pop(
                'type', None)):
            raise SolverConfigException(
                "Hyper optimizer type was not provided")
        elif not (hyper_optimizer_class := OPTIMIZERS_BY_NAME.get(
                hyper_optimizer_type, None)):
            raise SolverConfigException(
                f"There is no {hyper_optimizer_type} hyper optimizer type")
        else:
            hyper_optimizer = hyper_optimizer_class(**hyper_optimizer_config)

        params_inits = config.pop('params_inits', None)

        return cls(problem, pqc, optimizer, hyper_optimizer, params_inits)

    def solve(self, params_inits: dict[str, Any] = None) -> SolverResult:
        params_inits = params_inits or self.params_inits
        hyper_args = self.pqc.get_hopt_args(params_inits)

        if self.hyper_optimizer:
            wrapper = GlobalOptimizerFunction(
                self.pqc, self.problem, self.optimizer, params_inits)
            res = self.hyper_optimizer.minimize(wrapper, hyper_args)
            best_hargs = res.params
            local_opt_args = next(
                x for x in res.history[-1] if x.value == res.value)
            local_opt_args = next(
                x for x in local_opt_args.history[-1] if x.value == res.value
            ).params

            return SolverResult(
                self.pqc.run_with_probs(
                    self.problem, local_opt_args, best_hargs),
                self.pqc.get_params_init_format(
                    local_opt_args, best_hargs),
                res.history,
            )
        else:
            best_hargs = hyper_args

        opt_args = self.pqc.get_opt_args(params_inits, hyper_args=best_hargs)
        opt_wrapper = LocalOptimizerFunction(
                self.pqc, self.problem, best_hargs)
        opt_res = self.optimizer.minimize(opt_wrapper, opt_args)

        return SolverResult(
            self.pqc.run_with_probs(self.problem, opt_res.params, best_hargs),
            self.pqc.get_params_init_format(opt_res.params, best_hargs),
            opt_res.history,
        )
