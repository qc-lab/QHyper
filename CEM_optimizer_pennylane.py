from mimetypes import init
import numpy as np

from collections import deque
from typing import Dict, Any, Union, Callable, Optional, Tuple, List

POINT = Union[float, np.ndarray]


class CEMOptimizer:
    def __init__(self, init_params):
        # self.x = 0
        self.params = init_params
        self.mean = init_params
        self.cov = np.identity(len(init_params))
        self.best_weight = init_params

        super().__init__()

    def get_support_level(self):
        return {
            "gradient": False,
            "bounds": False,
            "initial_point": True
        }

    def step(
        self,
        fun: Callable[[POINT], float],
    ) -> list[list[float]]:
        n_iterations=10
        print_every=5
        pop_size=50
        elite_frac=0.2

        n_elite=int(pop_size*elite_frac)

        scores_deque = deque(maxlen=100)
        scores = []

        for i_iteration in range(1, n_iterations+1):
            points = np.random.multivariate_normal(self.mean, self.cov, size=pop_size)
            points = np.concatenate((points, [self.best_weight]), axis=0)
            # ys = np.random.normal(y_mean, 0.04, size=50)
            # print(weights_pop)
            rewards = np.array([fun(point) for point in points])

            elite_idxs = rewards.argsort()[:n_elite]
            elite_weights = [points[i] for i in elite_idxs]

            # print(elite_weights)
            best_weight = elite_weights[0]

            reward = fun(best_weight)
            if reward < fun(self.best_weight):
                self.best_weight = best_weight
            scores_deque.append(reward)
            scores.append(reward)
            self.mean = np.mean(elite_weights, axis=0)
            self.cov = np.cov(np.stack((elite_weights), axis = 1))

            if i_iteration % print_every == 0:
                # print(self.mean)
                # print(self.cov)
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
                # print(best_weight)
                print(f'{self.best_weight} with reward: {fun(self.best_weight)}')
        # calculate(best_weight, True)
        # return scores, best_weight
        # result = OptimizerResult
        # result.x(best_weight)
        # self.x = best_weight
        return self.best_weight