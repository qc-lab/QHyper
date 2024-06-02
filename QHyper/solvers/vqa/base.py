# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from dataclasses import dataclass
import numpy as np
from typing import Any
import numpy.typing as npt

from QHyper.problems.base import Problem

from QHyper.optimizers import (
    OPTIMIZERS, Dummy, Optimizer, OptimizationResult)
from QHyper.solvers.vqa.pqc.base import PQC

from QHyper.solvers.base import (
    Solver, SolverResult, SolverConfigException, SolverException)
from QHyper.solvers.vqa.pqc import PQC_BY_NAME


@dataclass
class LocalOptimizerFunction:
    pqc: PQC
    problem: Problem
    hyper_params: npt.NDArray

    def __call__(self, args: npt.NDArray) -> OptimizationResult:
        return self.pqc.run_opt(self.problem, args, self.hyper_params)


@dataclass
class GlobalOptimizerFunction:
    pqc: PQC
    problem: Problem
    optimizer: Optimizer
    params_config: dict[str, Any]

    def __call__(self, hargs: npt.NDArray) -> OptimizationResult:
        opt_args = self.pqc.get_opt_args(self.params_config, hyper_args=hargs)
        opt_wrapper = LocalOptimizerFunction(self.pqc, self.problem, hargs)
        return self.optimizer.minimize(opt_wrapper, opt_args)


class VQA(Solver):
    """
    Variational Quantum Algorithm solver.

    Attributes
    ----------

    problem : Problem
        The problem to be solved.
    pqc : PQC
        The parameterized quantum circuit. It should be a subclass of
        :py:class:`.PQC`.
    optimizer : Optimizer, default :py:class:`.Dummy`
        The optimizer to be used. It should be a subclass of
        :py:class:`.Optimizer`.
    hyper_optimizer : Optimizer, optional
        The hyper optimizer to be used. It should be a subclass of
        :py:class:`.Optimizer`.
    params_inits : dict[str, Any], optional
        Initial parameters for the solver. They might be overwritten by the
        parameters provided in the :py:meth:`.solve` method.
    """

    problem: Problem
    pqc: PQC
    optimizer: Optimizer
    hyper_optimizer: Optimizer | None
    params_inits: dict[str, Any] | None

    def __init__(
            self, problem: Problem, pqc: PQC, optimizer: Optimizer = Dummy(),
            hyper_optimizer: Optimizer | None = None,
            params_inits: dict[str, Any] | None = None
    ) -> None:
        self.problem = problem
        self.pqc = pqc
        self.optimizer = optimizer
        self.hyper_optimizer = hyper_optimizer
        self.params_inits = params_inits

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'VQA':
        error_msg = ""
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
        elif not (optimizer_class := OPTIMIZERS.get(
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
        elif not (hyper_optimizer_class := OPTIMIZERS.get(
                hyper_optimizer_type, None)):
            raise SolverConfigException(
                f"There is no {hyper_optimizer_type} hyper optimizer type")
        else:
            hyper_optimizer = hyper_optimizer_class(**hyper_optimizer_config)

        params_inits = config.pop('params_inits', None)

        return cls(problem, pqc, optimizer, hyper_optimizer, params_inits)

    def _find_best_result_from_history(
            self, histories: list[list[OptimizationResult]], best_value: float
    ) -> OptimizationResult:
        for history in reversed(histories):
            for result in history:
                if result.value == best_value:
                    return result
        raise SolverException(
            f"Could not find the result with value {best_value}")

    def solve(self, params_inits: dict[str, Any] = {}) -> SolverResult:
        if not params_inits:
            if not self.params_inits:
                raise SolverException("Params were not provided")
            params_inits = self.params_inits

        hyper_args = self.pqc.get_hopt_args(params_inits)

        _hyper_args = np.array(hyper_args).flatten()

        if self.hyper_optimizer:
            wrapper = GlobalOptimizerFunction(
                self.pqc, self.problem, self.optimizer, params_inits)
            res = self.hyper_optimizer.minimize(wrapper, _hyper_args)
            best_hargs = res.params

            global_results = self._find_best_result_from_history(
                res.history, res.value)
            local_opt_args = self._find_best_result_from_history(
                global_results.history, res.value
            ).params

            return SolverResult(
                self.pqc.run_with_probs(
                    self.problem, local_opt_args, best_hargs),
                self.pqc.get_params_init_format(
                    local_opt_args, best_hargs),
                res.history,
            )
        else:
            best_hargs = _hyper_args

        opt_args = self.pqc.get_opt_args(params_inits, hyper_args=best_hargs)
        opt_wrapper = LocalOptimizerFunction(
                self.pqc, self.problem, best_hargs)
        opt_res = self.optimizer.minimize(opt_wrapper, opt_args)

        return SolverResult(
            self.pqc.run_with_probs(self.problem, opt_res.params, best_hargs),
            self.pqc.get_params_init_format(opt_res.params, best_hargs),
            opt_res.history,
        )
