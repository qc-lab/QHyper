import multiprocessing as mp
import numpy as np

from .optimizer import HyperparametersOptimizer

class Random(HyperparametersOptimizer):
    epochs: int = 10
    samples_per_epoch: int = 100
    elite_frac: float = 0.1
    process: int = mp.cpu_count()

    def minimize(
        self, 
        func_creator, optimizer,
        init: list[float],
        bounds: list[float] = [-10, 10],
    ):
        best_weight = init
        best_reward = float('inf')

        def func(points):
            _func = func_creator(points)
            params = optimizer.minimize(_func, init)
            return _func(params)

        print_every=1

        for i_iteration in range(1, self.epochs+1):
            points_A = (bounds[1] - bounds[0]) * np.random.random(self.samples_per_epoch) + bounds[0]
            points_B = (bounds[1] - bounds[0]) * np.random.random(self.samples_per_epoch) + bounds[0]

            points = list(zip(points_A, points_B))

            points = np.concatenate((points, [best_weight]), axis=0)
            points = [list(point) for point in points]
            rewards = []
            # rewards = np.array(
            #     [self.problem.run_learning_n_get_results(list(point)) for point in points])

            # with mp.Pool(processes=self.process) as p:
            #     results = p.map(func, points)
            
            results = [func(point) for point in points]
            
            rewards = np.array([result for result in results])

            elite_idxs = rewards.argsort()[:self.n_elite]
            elite_weights = [points[i] for i in elite_idxs]

            best_weight = elite_weights[0]

            reward = func(best_weight)
            if reward < best_reward:
                best_weight = best_weight
                best_reward = reward

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
