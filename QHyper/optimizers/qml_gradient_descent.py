# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Callable, Any, Type

import pennylane as qml
from pennylane import numpy as np

from QHyper.optimizers.base import (
    Optimizer, OptimizationResult, OptimizerError, OptimizationParameter)


QML_GRADIENT_DESCENT_OPTIMIZERS: dict[
        str, Type[qml.GradientDescentOptimizer]] = {
    'adam': qml.AdamOptimizer,
    'adagrad': qml.AdagradOptimizer,
    'rmsprop': qml.RMSPropOptimizer,
    'momentum': qml.MomentumOptimizer,
    'nesterov_momentum': qml.NesterovMomentumOptimizer,
    'sgd': qml.GradientDescentOptimizer,
    'qng': qml.QNGOptimizer,
}


class QmlGradientDescent(Optimizer):
    """Gradient Descent Optimizer

    This minimizer is a wrapper for gradient descent optimizers
    provided by PennyLane.
    This alogrithm requries the following parameters to be set:
    - `init` initial values for each parameter

    Attributes
    ----------
    optimizer : qml.GradientDescentOptimizer
        object of class GradientDescentOptimizer or inheriting from this class
    steps : int, default 200
        number of optimization steps
    stepsize : float, default 0.005
        stepsize for the optimizer
    verbose : bool, default False
        if set to True, additional information will be printed
    """

    optimizer: qml.GradientDescentOptimizer
    steps: int
    stepsize: float
    verbose: bool

    def __init__(
        self,
        name: str = 'adam',
        steps: int = 200,
        stepsize: float = 0.005,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        name : str, default 'adam'
            name of the gradient descent optimizer provided by PennyLane
        steps : int, default 200
            number of optimization steps
        stepsize : float, default 0.005
            stepsize for the optimizer
        verbose : bool, default False
            if set to True, additional information will be printed
        **kwargs : Any
            Additional arguments that will be passed to the PennyLane
            optimizer. More infomation can be found in the PennyLane
            documentation.
        """
        if name not in QML_GRADIENT_DESCENT_OPTIMIZERS:
            raise ValueError(
                f'Optimizer {name} not found. '
                'Available optimizers: '
                f'{list(QML_GRADIENT_DESCENT_OPTIMIZERS.keys())}'
            )

        self.optimizer = QML_GRADIENT_DESCENT_OPTIMIZERS[name](
            stepsize=stepsize,
            **kwargs
        )
        self.steps = steps
        self.verbose = verbose

    def minimize(
        self,
        func: Callable[[list[float]], OptimizationResult],
        init: OptimizationParameter
    ) -> OptimizationResult:
        init.assert_init()
        if isinstance(self.optimizer, qml.QNGOptimizer):
            raise OptimizerError(
                'QNG is not supported via optimizer, use qml_qaoa instead')

        def wrapper(params: list[float]) -> float:
            return func(list(params)).value

        cost_history = []
        best_result = float('inf')
        best_params = np.array(init.init, requires_grad=True)
        params = np.array(init.init, requires_grad=True)

        if hasattr(self.optimizer, 'reset'):
            self.optimizer.reset()  # type: ignore
        for i in range(self.steps):
            params, cost = self.optimizer.step_and_cost(wrapper, params)
            params = np.array(params, requires_grad=True)

            if cost < best_result:
                best_params = params
                best_result = cost
            cost_history.append(OptimizationResult(float(cost), params))

            if self.verbose:
                print(f'Step {i+1}/{self.steps}: {float(cost)}')

        return OptimizationResult(best_result, best_params, [cost_history])

    def minimize_expval_func(
            self, func: qml.QNode, init: OptimizationParameter
    ) -> OptimizationResult:
        """
        Used in :py:class:`.QML_QAOA` to minimize the
        expectation value function.
        """

        cost_history = []
        cost = float('inf')
        params = np.array(init.init, requires_grad=True)
        if hasattr(self.optimizer, 'reset'):
            self.optimizer.reset()  # type: ignore
        for i in range(self.steps):
            params, cost = self.optimizer.step_and_cost(func, params)
            params = np.array(params, requires_grad=True)

            cost_history.append(OptimizationResult(float(cost), params))

            if self.verbose:
                print(f'Step {i+1}/{self.steps}: {float(cost)}')

        return OptimizationResult(cost, params, [cost_history])
