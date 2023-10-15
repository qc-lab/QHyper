"""
Refactored from scipy.optimize.basinhopping
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

basinhopping: The basinhopping global optimization algorithm
"""
import numpy as np
import math
import inspect
import scipy.optimize
from scipy._lib._util import check_random_state

from typing import Callable, Union
import numpy.typing as npt

from .base import Optimizer, OptimizationResult


_params = (
    inspect.Parameter("res_new", kind=inspect.Parameter.KEYWORD_ONLY),
    inspect.Parameter("res_old", kind=inspect.Parameter.KEYWORD_ONLY),
)
_new_accept_test_signature = inspect.Signature(parameters=_params)


class Storage:
    """
    Class used to store the lowest energy structure
    """

    def __init__(self, minres):
        self._add(minres)

    def _add(self, minres):
        self.minres = minres
        self.minres.params = np.copy(minres.params)

    def update(self, minres):
        if minres.value < self.minres.value:
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        return self.minres


class BasinHoppingRunner:
    """This class implements the core of the basinhopping algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly. Signature should
        be ``x_new = step_taking(x)``. Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`. These tests will be used to judge whether or not to accept
        the step. The acceptable return values are True, False, or ``"force
        accept"``. If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``basinhopping`` is trapped in.
    disp : bool, optional
        Display status messages.

    """

    def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.disp = disp

        self.nstep = 0

        # initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.minimization_failures = 0

        # do initial minimization
        minres = minimizer(self.x)
        print(minimizer)
        # if not minres.success:
        #     self.res.minimization_failures += 1
        #     if self.disp:
        #         print("warning: basinhopping: local minimization failure")
        # self.x = np.copy(minres.params)
        self.energy = minres.value
        self.incumbent_minres = minres  # best minimize result found so far
        if self.disp:
            print("basinhopping step %d: f %g" % (self.nstep, self.energy))

        # initialize storage class
        self.storage = Storage(minres)

        # if hasattr(minres, "nfev"):
        #     self.res.nfev = minres.nfev
        # if hasattr(minres, "njev"):
        #     self.res.njev = minres.njev
        # if hasattr(minres, "nhev"):
        #     self.res.nhev = minres.nhev

    def _monte_carlo_step(self):
        """Do one Monte Carlo iteration

        Randomly displace the coordinates, minimize, and decide whether
        or not to accept the new coordinates.
        """
        # Take a random step.  Make a copy of x because the step_taking
        # algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        # do a local minimization
        minres = self.minimizer(x_after_step)
        # x_after_quench = minres.params
        energy_after_quench = minres.value
        # if not minres.success:
        #     self.res.minimization_failures += 1
        #     if self.disp:
        #         print("warning: basinhopping: local minimization failure")
        if hasattr(minres, "nfev"):
            self.res.nfev += minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev += minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev += minres.nhev

        # accept the move based on self.accept_tests. If any test is False,
        # then reject the step.  If any test returns the special string
        # 'force accept', then accept the step regardless. This can be used
        # to forcefully escape from a local minimum if normal basin hopping
        # steps are not sufficient.
        accept = True
        for test in self.accept_tests:
            if inspect.signature(test) == _new_accept_test_signature:
                testres = test(res_new=minres, res_old=self.incumbent_minres)
            else:
                testres = test(
                    f_new=energy_after_quench,
                    x_new=x_after_step,
                    f_old=self.energy,
                    x_old=self.x,
                )

            if testres == "force accept":
                accept = True
                break
            elif testres is None:
                raise ValueError(
                    "accept_tests must return True, "
                    "False, " "or " "'force accept'"
                )
            elif not testres:
                accept = False

        # Report the result of the acceptance test to the take step class.
        # This is for adaptive step taking
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(
                accept,
                f_new=energy_after_quench,
                x_new=x_after_step,
                f_old=self.energy,
                x_old=self.x,
            )

        return accept, minres, x_after_step

    def one_cycle(self):
        """Do one cycle of the basinhopping algorithm"""
        self.nstep += 1
        new_global_min = False

        accept, minres, x_after_step = self._monte_carlo_step()

        if accept:
            self.energy = minres.value
            # self.x = np.copy(minres.params)
            self.x = x_after_step
            self.incumbent_minres = minres  # best minimize result found so far
            new_global_min = self.storage.update(minres)

        # print some information
        if self.disp:
            self.print_report(minres.value, accept)
            if new_global_min:
                print(
                    "found new global minimum on step %d with function"
                    " value %g" % (self.nstep, self.energy)
                )

        # save some variables as BasinHoppingRunner attributes
        self.xtrial = x_after_step
        self.energy_trial = minres.value
        self.accept = accept

        return new_global_min

    def print_report(self, energy_trial, accept):
        """print a status update"""
        minres = self.storage.get_lowest()
        print(
            "basinhopping step %d: f %g trial_f %g accepted %d "
            " lowest_f %g"
            % (self.nstep, self.energy, energy_trial, accept, minres.value)
        )


class AdaptiveStepsize:
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """

    def __init__(
        self, takestep, accept_rate=0.5, interval=50, factor=0.9, verbose=True
    ):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)

    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            # We're accepting too many steps. This generally means we're
            # trapped in a basin. Take bigger steps.
            self.takestep.stepsize /= self.factor
        else:
            # We're not accepting enough steps. Take smaller steps.
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print(
                "adaptive stepsize: acceptance rate {:f} target {:f} new "
                "stepsize {:g} old stepsize {:g}".format(
                    accept_rate,
                    self.target_accept_rate,
                    self.takestep.stepsize,
                    old_stepsize,
                )
            )

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        "called by basinhopping to report the result of the step"
        if accept:
            self.naccept += 1


class RandomDisplacement:
    """Add a random displacement of maximum size `stepsize` to each coordinate.

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """

    def __init__(self, stepsize=0.5, random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)

    def __call__(self, x):
        x += self.random_gen.uniform(
            -self.stepsize, self.stepsize, np.shape(x))
        return x


# class MinimizerWrapper:
#     """
#     wrap a minimizer function as a minimizer class
#     """
#     def __init__(self, minimizer, func=None, **kwargs):
#         self.minimizer = minimizer
#         self.func = func
#         self.kwargs = kwargs

#     def __call__(self, x0):
#         if self.func is None:
#             return self.minimizer(x0, **self.kwargs)
#         else:
#             return self.minimizer(self.func, x0, **self.kwargs)


class Metropolis:
    """Metropolis acceptance criterion.

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Random number generator used for acceptance test.

    """

    def __init__(self, T, random_gen=None):
        # Avoid ZeroDivisionError since "MBH can be regarded as a special case
        # of the BH framework with the Metropolis criterion, where temperature
        # T = 0." (Reject all steps that increase energy.)
        self.beta = 1.0 / T if T != 0 else float("inf")
        self.random_gen = check_random_state(random_gen)

    def accept_reject(self, res_new, res_old):
        """
        Assuming the local search underlying res_new was successful:
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted,
        less likely for larger differences.
        """
        with np.errstate(invalid="ignore"):
            # The energy values being fed to Metropolis are 1-length arrays,
            # so we need to add types to avoid warnings.
            # and if they are equal, their difference is 0, which gets
            # multiplied by beta, which is inf, and array([0]) * float('inf')
            # causes
            #
            # RuntimeWarning: invalid value encountered in multiply
            #
            # Ignore this warning so when the algorithm is on a flat plane,
            # it always accepts the step, to try to move off the plane.
            prod = -(res_new.value - res_old.value) * self.beta
            w = math.exp(min(0, prod))

        rand = self.random_gen.uniform()
        return w >= rand  # and (res_new.success or not res_old.success)

    def __call__(self, *, res_new, res_old):
        """
        f_new and f_old are mandatory in kwargs
        """
        return bool(self.accept_reject(res_new, res_old))


class Basinhopping(Optimizer):
    def __init__(self, niter: int = 100) -> None:
        self.niter = niter

    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], OptimizationResult],
        init: npt.NDArray[np.float64],
        T: float = 1.0,
        stepsize: float = 0.5,
        take_step: Callable[
            [npt.NDArray[np.float64], float], npt.NDArray[np.float64]] = None,
        accept_test: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], bool] = None,
        callback: Union[
            Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], None],
            list[Callable[
                [npt.NDArray[np.float64], npt.NDArray[np.float64]], None]]
            ] = None,
        interval: int = 50,
        disp: bool = False,
        niter_success: int = None,
        seed: int = None,
        *,
        target_accept_rate: float = 0.5,
        stepwise_factor: float = 0.9
    ) -> OptimizationResult:
        if target_accept_rate <= 0.0 or target_accept_rate >= 1.0:
            raise ValueError("target_accept_rate has to be in range (0, 1)")
        if stepwise_factor <= 0.0 or stepwise_factor >= 1.0:
            raise ValueError("stepwise_factor has to be in range (0, 1)")

        x0 = np.array(init)

        # set up the np.random generator
        rng = check_random_state(seed)

        # set up minimizer
        # if minimizer_kwargs is None:
        # minimizer_kwargs = dict()
        # wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
        #                                     **minimizer_kwargs)

        # set up step-taking algorithm
        if take_step is not None:
            if not callable(take_step):
                raise TypeError("take_step must be callable")
            # if take_step.stepsize exists then use AdaptiveStepsize to control
            # take_step.stepsize
            if hasattr(take_step, "stepsize"):
                take_step_wrapped = AdaptiveStepsize(
                    take_step,
                    interval=interval,
                    accept_rate=target_accept_rate,
                    factor=stepwise_factor,
                    verbose=disp,
                )
            else:
                take_step_wrapped = take_step
        else:
            # use default
            displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
            take_step_wrapped = AdaptiveStepsize(
                displace,
                interval=interval,
                accept_rate=target_accept_rate,
                factor=stepwise_factor,
                verbose=disp,
            )

        # set up accept tests
        accept_tests = []
        if accept_test is not None:
            if not callable(accept_test):
                raise TypeError("accept_test must be callable")
            accept_tests = [accept_test]

        # use default
        metropolis = Metropolis(T, random_gen=rng)
        accept_tests.append(metropolis)

        if niter_success is None:
            niter_success = self.niter + 2

        bh = BasinHoppingRunner(
            x0, func, take_step_wrapped, accept_tests, disp=disp)

        # The wrapped minimizer is called once during construction of
        # BasinHoppingRunner, so run the callback
        if callable(callback):
            callback(bh.storage.minres.params, bh.storage.minres.value, True)

        # start main iteration loop
        count, i = 0, 0
        message = [
            "requested number of basinhopping iterations completed"
            " successfully"
        ]
        for i in range(self.niter):
            new_global_min = bh.one_cycle()

            if callable(callback):
                # should we pass a copy of x?
                val = callback(bh.xtrial, bh.energy_trial, bh.accept)
                if val is not None:
                    if val:
                        message = [
                            "callback function requested stop early by"
                            "returning True"
                        ]
                        break

            count += 1
            if new_global_min:
                count = 0
            elif count > niter_success:
                message = ["success condition satisfied"]
                break

        # prepare return object
        res = bh.res
        res.lowest_optimization_result = bh.storage.get_lowest()
        # res.x = np.copy(res.lowest_optimization_result.)
        res.fun = res.lowest_optimization_result.value
        res.message = message
        res.nit = i + 1
        # res.success = res.lowest_optimization_result.success
        return OptimizationResult(res.fun, bh.x)
