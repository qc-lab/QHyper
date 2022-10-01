import numpy as np

from dataclasses import dataclass, field

import multiprocessing as mp

from ..problems.problem import Problem
from ..problem_solver import ProblemSolver
import scipy


@dataclass
class CEM:
    solver: ProblemSolver
    epochs: int = 10
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    n_elite: int = field(init=False)
    process: int = mp.cpu_count()

    def __post_init__(self):
        self.n_elite = int(self.samples_per_epoch * self.elite_frac)

    def minimize(
        self, 
        init_weights: list[float],
        mean: list[float] | None = None,
        cov: np.ndarray | None = None
    ):
        _mean = [1] * len(init_weights) if mean is None else mean
        _cov = np.identity(len(init_weights)) if cov is None else cov
        best_weight = init_weights
        best_reward = float('inf')

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

            points = np.concatenate((points, [best_weight]), axis=0)
            points = [list(point) for point in points]
            rewards = []
            # rewards = np.array(
            #     [self.problem.run_learning_n_get_results(list(point)) for point in points])

            with mp.Pool(processes=self.process) as p:
                results = p.map(self.solver.get_score, points)
            
            rewards = np.array([result for result in results])

            elite_idxs = rewards.argsort()[:self.n_elite]
            elite_weights = [points[i] for i in elite_idxs]

            best_weight = elite_weights[0]

            reward = self.solver.get_score(best_weight)
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
