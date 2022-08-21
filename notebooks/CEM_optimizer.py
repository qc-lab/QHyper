import numpy as np

from collections import deque
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult
from typing import Dict, Any, Union, Callable, Optional, Tuple, List

POINT = Union[float, np.ndarray]


class CEMOptimizer(Optimizer):
    def __init__(self, **args):
        print(args)
        # self.x = 0
        super().__init__()

    def get_support_level(self):
        return {
            "gradient": False,
            "bounds": False,
            "initial_point": True
        }

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        n_iterations=30
        print_every=5
        pop_size=50
        elite_frac=0.2

        n_elite=int(pop_size*elite_frac)

        scores_deque = deque(maxlen=100)
        scores = []
        best_weight = x0
        mean = [1] * len(x0)
        cov = np.zeros((len(x0), len(x0))) 
        for i_iteration in range(1, n_iterations+1):
            points = np.random.multivariate_normal(mean, cov, size=pop_size)
            points = np.concatenate((points, [best_weight]), axis=0)
            # ys = np.random.normal(y_mean, 0.04, size=50)
            # print(weights_pop)
            rewards = np.array([fun(point) for point in points])

            elite_idxs = rewards.argsort()[:n_elite]
            elite_weights = [points[i] for i in elite_idxs]

            # print(elite_weights)
            best_weight = elite_weights[0]

            reward = fun(best_weight)
            scores_deque.append(reward)
            scores.append(reward)
            mean = np.mean(elite_weights, axis=0)
            cov = np.cov(np.stack((elite_weights), axis = 1))

            if i_iteration % print_every == 0:
                print(mean)
                print(cov)
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
                print(f'{best_weight} with reward: {fun(best_weight, True)}')
        # calculate(best_weight, True)
        # return scores, best_weight
        result = OptimizerResult
        result.x(best_weight)
        self.x = best_weight
        return result