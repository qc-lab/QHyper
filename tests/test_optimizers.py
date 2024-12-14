import numpy as np
import pytest

from QHyper.optimizers.cem import CEM
from QHyper.optimizers.grid_search import GridSearch
from QHyper.optimizers.random import Random
from QHyper.optimizers.scipy_minimizer import ScipyOptimizer
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.optimizers.base import OptimizationResult, OptimizationParameter

np.random.seed(1244)


def function(args) -> OptimizationResult:
    x, y, z = args
    return OptimizationResult((x + y + z)**2, np.array([x, y, z]), [[]])


def test_scipy():
    minimizer = ScipyOptimizer(
        maxfun=100,
        method='L-BFGS-B'
    )
    init = OptimizationParameter(
        init=[1., 0.5, -0.3], min=[-1, -1, -1], max=[1, 1, 1])
    result = minimizer.minimize(function, init)
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_qml():
    minimizer = QmlGradientDescent()
    init = OptimizationParameter(init=[1., 0.5, -0.3])
    result = minimizer.minimize(function, init)
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_random():
    minimizer = Random(
        processes=1,
        number_of_samples=1000,
        disable_tqdm=True,
    )
    init = OptimizationParameter(
        init=[1., 0.5, -0.3], min=[-1, -1, -1], max=[1, 1, 1])
    result = minimizer.minimize(function, init)
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_cem():
    minimizer = CEM(
        processes=1,
        samples_per_epoch=100,
        epochs=3,
        disable_tqdm=True,
    )
    init = OptimizationParameter(
        init=[1., 0.5, -0.3], min=[-1, -1, -1], max=[1, 1, 1])
    result = minimizer.minimize(function, init)
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)


def test_grid():
    minimizer = GridSearch(
        processes=1,
        disable_tqdm=True,
    )
    init = OptimizationParameter(
        step=[0.5, 0.5, 0.5], min=[-1, -1, -1], max=[1, 1, 1])
    result = minimizer.minimize(function, init)
    assert result.value == pytest.approx(0, rel=1e-6, abs=1e-6)
