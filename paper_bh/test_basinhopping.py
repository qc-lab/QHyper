import argparse
import multiprocessing as mp

import numpy as np

from QHyper.problems.knapsack import KnapsackProblem
from QHyper.solvers.vqa.base import VQA


BASINHOPPING_HQAOA_CONFIG = {
    'optimizer': {
        'type': 'basinhopping',
        'maxfun': 200,
        'niter': None,
        'bounds': [(1, 10)]*3 + [(0, 2*np.pi)]*10
    },
    'pqc': {
        'type': 'hqaoa',
        'layers': 5,
    },
}

BASINHOPPING_WFQAOA_CONFIG = {
    'optimizer': {
        'type': 'basinhopping',
        'maxfun': 200,
        'niter': None,
        'bounds': [(0, 2*np.pi)]*10
    },
    'pqc': {
        'type': 'wfqaoa',
        'layers': 5
    },
}

TESTER_CONFIG = {
    'pqc': {
        'type': 'wfqaoa',
        'layers': 5,
    }
}


class Wrapper:
    def __init__(self, max_weight, items, samples, output_file, niter) -> None:
        knapsack = KnapsackProblem(
            max_weight=max_weight, items=items)

        BASINHOPPING_WFQAOA_CONFIG['optimizer']['niter'] = niter
        BASINHOPPING_HQAOA_CONFIG['optimizer']['niter'] = niter
        self.bh_wfqaoa = VQA(knapsack, config=BASINHOPPING_WFQAOA_CONFIG)
        self.bh_hqaoa = VQA(knapsack, config=BASINHOPPING_HQAOA_CONFIG)
        self.tester = VQA(knapsack, config=TESTER_CONFIG)

        self.angles = 2 * np.pi * np.random.rand(samples, 2, 5)
        self.weights = 10*np.random.rand(samples, 2) + 1
        self.output_file = output_file

        with open(output_file, mode='a') as output:
            print(f"{knapsack.knapsack.items} {knapsack.knapsack.max_weight} niter: {niter}", file=output)
            print("BH_WFQAOA;BH_HQAOA", file=output)

    def __call__(self, i) -> None:
        with open(self.output_file, mode='a') as output:
            params_cofing = {
                'angles': self.angles[i],
                'hyper_args': list(self.weights[i]) + [self.weights[i][1]]
            }
            for solver in [self.bh_wfqaoa, self.bh_hqaoa]:
                best_params = solver.solve(params_cofing)

                print(f"{self.tester.evaluate(best_params):.3f}", end=';', file=output)
            print(file=output)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('-w', type=int)
    parser.add_argument('--weights', metavar='N', type=int, nargs='+')
    parser.add_argument('--values', metavar='N', type=int, nargs='+')
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--niter', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    if len(args.weights) != len(args.values):
        raise Exception("Number of provided weights and values must be the same")
    items=list(zip(args.weights, args.values))
    output_file = f'bh_{args.niter}_{args.w}_{"_".join([f"{w}-{v}" for w,v in list(zip(args.weights, args.values))])}.csv'
    wrapper = Wrapper(args.w, items, args.samples, output_file, args.niter)
   
    # qaoa = VQA(knapsack, config=QAOA_CONFIG)
    # wfqaoa = VQA(knapsack, config=WFQAOA_CONFIG)
    # hqaoa = VQA(knapsack, config=HQAOA_CONFIG)
    # tester = VQA(knapsack, config=TESTER_CONFIG)
        
    with mp.Pool(processes=args.processes) as pool:
        pool.map(wrapper, range(args.samples))
