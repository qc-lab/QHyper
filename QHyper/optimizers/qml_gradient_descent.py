from typing import Callable, Any
import numpy.typing as npt

import pennylane as qml
from pennylane import numpy as np

from .base import Optimizer, OptimizationResult


QML_GRADIENT_DESCENT_OPTIMIZERS: dict[str, qml.GradientDescentOptimizer] = {
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

    Using GradientDescentOptimizer from library PennyLane.

    Attributes
    ----------
    optimizer : qml.GradientDescentOptimizer
        object of class GradientDescentOptimizer or inheriting from this class
    steps : int
        number of optimization steps
    stepsize : float
        stepsize for the optimizer
    verbose : bool
        if set to True, additional information will be printed (default False)
    **kwargs : Any
        additional arguments for the optimizer
    """

    def __init__(
        self,
        optimizer: str = 'adam',
        steps: int = 200,
        stepsize: float = 0.005,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        optimizer : str
            name of the gradient descent optimizer provided by PennyLane
        steps : int
            number of optimization steps
        stepsize : float
            stepsize for the optimizer
        """
        if optimizer not in QML_GRADIENT_DESCENT_OPTIMIZERS:
            raise ValueError(
                f'Optimizer {optimizer} not found. '
                'Available optimizers: '
                f'{list(QML_GRADIENT_DESCENT_OPTIMIZERS.keys())}'
            )

        self.optimizer = QML_GRADIENT_DESCENT_OPTIMIZERS[optimizer](
            stepsize=stepsize,
            **kwargs
        )
        self.steps = steps
        self.verbose = verbose

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
        init: npt.NDArray[np.float64]
    ) -> OptimizationResult:
        """Returns params which lead to the lowest value of
            the provided function and cost history

        Parameters
        ----------
        func : Callable[[ArgsType], float]
            function which will be minimized
        init : ArgsType
            initial args for optimizer

        Returns
        -------
        tuple[ArgsType, Any]
            Returns tuple which contains params taht lead to the lowest value
            of the provided function and cost history
        """
        if isinstance(self.optimizer, qml.QNGOptimizer):
            raise ValueError(
                'QNG is not supported via optimizer, use qml_qaoa instead')

        def wrapper(params: npt.NDArray[np.float64]) -> float:
            return func(np.array(params).reshape(np.array(init).shape)).value

        cost_history = []
        params = np.array(init, requires_grad=True)
        if "reset" in dir(self.optimizer):
            self.optimizer.reset()
        for i in range(self.steps):
            params, cost = self.optimizer.step_and_cost(wrapper, params)
            cost_history.append(OptimizationResult(float(cost), params))

            if self.verbose:
                print(f'Step {i+1}/{self.steps}: {float(cost)}')

        return OptimizationResult(cost, params, cost_history)

    def minimize_expval_func(
            self, func: qml.QNode, init: npt.NDArray[np.float64]
    ) -> OptimizationResult:

        cost_history = []
        params = np.array(init, requires_grad=True)
        if "reset" in dir(self.optimizer):
            self.optimizer.reset()
        for i in range(self.steps):
            params, cost = self.optimizer.step_and_cost(func, params)
            cost_history.append(OptimizationResult(float(cost), params))

            if self.verbose:
                print(f'Step {i+1}/{self.steps}: {float(cost)}')

        return OptimizationResult(cost, params, [cost_history])
