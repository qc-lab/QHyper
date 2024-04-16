# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from typing import Any, Optional

from QHyper.problems.base import Problem
from QHyper.optimizers import OptimizationResult


PQCResults = tuple[dict[str, float], list[float]]


class PQC:
    """
    Abstract base class for Parameterized Quantum Circuit (PQC).

    Attributes
    ----------
    pqc_type : str
        Type of the parameterized quantum circuit.
    """

    pqc_type: str

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None: ...

    @abstractmethod
    def run_opt(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        """
        Run optimization using the PQC.

        Parameters
        ----------
        problem : Problem
            The problem to be solved.
        opt_args : npt.NDArray[np.float64]
            Optimization arguments.
        hyper_args : npt.NDArray[np.float64]
            Hyperparameter optimization arguments.

        Returns
        -------
        float
            The result of the optimization.
        """
        ...

    @abstractmethod
    def run_with_probs(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """
        Run optimization using the PQC and return probabilities.

        Parameters
        ----------
        problem : Problem
            The problem to be solved.
        opt_args : npt.NDArray[np.float64]
            Optimization arguments.
        hyper_args : npt.NDArray[np.float64]
            Hyperparameter optimization arguments.

        Returns
        -------
        dict[str, float]
            Probabilities of the results.
        """
        ...

    @abstractmethod
    def get_opt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Get optimization arguments. This method should return arguments for
        optimizer. These arguments may come from initial parameters
        (params_init) or might be override by args or hyper_args.

        Parameters
        ----------
        params_init : dict[str, Any]
            Initial parameters for the optimization.
        args : Optional[npt.NDArray[np.float64]], optional
            Additional arguments, by default None.
        hyper_args : Optional[npt.NDArray[np.float64]], optional
            Hyperparameter optimization arguments, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            Optimization arguments.
        """
        ...

    @abstractmethod
    def get_hopt_args(
        self,
        params_init: dict[str, Any],
        args: Optional[npt.NDArray[np.float64]] = None,
        hyper_args: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Get hyperparameter optimization arguments.
        This method should return arguments for hyperoptimizer.
        These arguments may come from initial parameters
        (params_init) or might be override by args or hyper_args.

        Parameters
        ----------
        params_init : dict[str, Any]
            Initial parameters for the optimization.
        args : Optional[npt.NDArray[np.float64]], optional
            Additional arguments, by default None.
        hyper_args : Optional[npt.NDArray[np.float64]], optional
            Hyperparameter optimization arguments, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            Hyperparameter optimization arguments.
        """
        ...

    @abstractmethod
    def get_params_init_format(
        self,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        """
        Get initial params format. Method changes opt_args and hyper_args into
        dict in same format at provided params_init in different methods.

        Parameters
        ----------
        opt_args : npt.NDArray[np.float64]
            Optimization arguments.
        hyper_args : npt.NDArray[np.float64]
            Hyperparameter optimization arguments.

        Returns
        -------
        dict[str, Any]
            Initial arguments.
        """
        ...
