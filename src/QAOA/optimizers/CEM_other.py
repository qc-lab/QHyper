import numpy as np

from dataclasses import dataclass, field

import multiprocessing as mp
import tqdm 
from typing import Callable, Any

from .optimizer import HyperparametersOptimizer, Worker, ArgsType, Optimizer


@dataclass
class CEM(HyperparametersOptimizer):
    epochs: int = 10
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    n_elite: int = field(init=False)
    process: int = mp.cpu_count()

    def __post_init__(self):
        self.n_elite = int(self.samples_per_epoch * self.elite_frac)

    def minimize(
        self, 
        func_creator: Callable[[ArgsType], Callable[[ArgsType], float]], 
        optimizer: Optimizer,
        init: ArgsType, 
        hyperparams_init: ArgsType = None, 
        bounds: list[float] = None,
    ) -> ArgsType:
        mean = [1] * len(hyperparams_init) 
        cov = np.identity(len(hyperparams_init))
        best_weight = hyperparams_init
        best_reward = float('inf')

        print_every=1

        # scores_deque = deque(maxlen=100)
        scores = []
        worker = Worker(func_creator, optimizer, init)
        # self.func_creator, self.optimizer, self.init = func_creator, optimizer, init
        for i_iteration in range(1, self.epochs+1):
            points = np.random.multivariate_normal(mean, cov, size=self.samples_per_epoch)
            points[points < 0] = 0
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
            with mp.Pool(processes=self.process) as p:
                results = list(tqdm.tqdm(p.imap(worker.func, points), total=len(points)))
            #     results = p.map(worker.func, points)
            # results = [worker.func(point) for point in points]
            # results = process_map(worker.func, points, max_workers=self.process)

            rewards = np.array([result for result in results])

            elite_idxs = rewards.argsort()[:self.n_elite]
            elite_weights = [points[i].flatten() for i in elite_idxs]

            best_weight = elite_weights[0].reshape(hyperparams_init.shape)
            # print(best_weight)
            reward = worker.func(best_weight)
            if reward < best_reward:
                best_weight = best_weight
                best_reward = reward
            # scores_deque.append(reward)
            scores.append(reward)
            mean = np.mean(elite_weights, axis=0)
            cov = np.cov(np.stack((elite_weights), axis = 1), bias=True)

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
