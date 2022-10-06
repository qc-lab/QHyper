from cgitb import reset
import numpy as np

from dataclasses import dataclass, field

import multiprocessing as mp
from typing import Callable, Any

# from ..QAOA_problems.problem import Problem
# from ..problem_solver import ProblemSolver
from .optimizer import Optimizer
from ..QAOA_problems.problem import Problem
import scipy


@dataclass
class CEM(Optimizer):
    # solver: ProblemSolver
    epochs: int = 10
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    n_elite: int = field(init=False)
    process: int = mp.cpu_count()

    def __post_init__(self):
        self.n_elite = int(self.samples_per_epoch * self.elite_frac)

    # def set_func_from_problem(self, problem: Problem, hyperparameters: dict[str, Any]):
    #     global wrapper
    #     def wrapper(params):
    #         probs = problem.get_probs_func(**hyperparameters)(params)
    #         return problem.check_results(probs)
    #     self.func = wrapper

    def minimize(
        self, 
        # func: Callable,
        func_creator, optimizer, init, hyperparams_init
        # mean: list[float] | None = None,
        # cov: np.ndarray | None = None
    ):
        # flatted_init = hyperparams_init.flatten()
        _mean = [1] * len(hyperparams_init) # if mean is None else mean
        _cov = np.identity(len(hyperparams_init)) # if cov is None else cov
        best_weight = hyperparams_init
        best_reward = float('inf')

        def func(points):
            _func = func_creator(points)
            params = optimizer.minimize(_func, init)
            return _func(params)
        print_every=1

        # scores_deque = deque(maxlen=100)
        scores = []

        for i_iteration in range(1, self.epochs+1):
            points = np.random.multivariate_normal(_mean, _cov, size=self.samples_per_epoch)

            # std_dev = np.full(self.samples_per_epoch, np.sqrt(_cov[1][1]))
            # mean_B = np.full(self.samples_per_epoch, _mean[1])
            # lower_bound_B = np.full(self.samples_per_epoch, 0)
            # upper_bound_B = np.full(self.samples_per_epoch, 100)
            # points_B = scipy.stats.truncnorm.rvs(
            #     (lower_bound_B - mean_B) / std_dev, 
            #     upper_bound_B, 
            #     loc=mean_B, 
            #     scale=std_dev
            # )

            # max_c = 2
            # mean_A = np.full(self.samples_per_epoch, _mean[0])
            # std_dev = np.full(self.samples_per_epoch, np.sqrt(_cov[0][0]))
            # points_A = scipy.stats.truncnorm.rvs(
            #     ((points_B*max_c + 0.1)-mean_A)/std_dev, 
            #     (points_B*max_c + 10)/std_dev, 
            #     loc=mean_A, 
            #     scale=std_dev
            # )

            # points = list(zip(points_A, points_B))

            points = np.concatenate((points, [best_weight.flatten()]), axis=0)
            points = [np.reshape(np.array(point), hyperparams_init.shape) for point in points]
            rewards = []
            # rewards = np.array(
            #     [self.problem.run_learning_n_get_results(list(point)) for point in points])

            # with mp.Pool(processes=self.process) as p:
            #     results = p.map(func, points)
            results = [func(point) for point in points]

            rewards = np.array([result for result in results])

            elite_idxs = rewards.argsort()[:self.n_elite]
            elite_weights = [points[i].flatten() for i in elite_idxs]

            best_weight = elite_weights[0].reshape(hyperparams_init.shape)
            # print(best_weight)
            reward = func(best_weight)
            if reward < best_reward:
                best_weight = best_weight
                best_reward = reward
            # scores_deque.append(reward)
            scores.append(reward)
            _mean = np.mean(elite_weights, axis=0)
            _cov = np.cov(np.stack((elite_weights), axis = 1), bias=True)

            if i_iteration % print_every == 0:
                # print(self.mean)
                # print(self.cov)
                print(f'Epoch {i_iteration}\t'
                      f'Average Elite Score: {np.average([rewards[i] for i in elite_idxs])}\t'
                      f'Average Score: {np.average(rewards)}'
                )
                # print(best_weight)
                print(f'{best_weight} with reward: {best_reward}')
        return best_weight
