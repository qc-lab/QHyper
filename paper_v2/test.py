import argparse

import numpy as np

from QHyper.problems.knapsack import KnapsackProblem
from QHyper.solvers.vqa.base import VQA


HQAOA_CONFIG = {
    'optimizer': {
        'type': 'scipy',
        'maxfun': 200,
        'bounds': [(1, 10)]*3+[(0, 2*np.pi)]*10
    },
    'pqc': {
        'type': 'hqaoa',
        'layers': 5,
    },
}

WFQAOA_CONFIG = {
    'optimizer': {
        'type': 'scipy',
        'maxfun': 200,
    },
    'pqc': {
        'type': 'wfqaoa',
        'layers': 5,
    },
}

QAOA_CONFIG = {
    'optimizer': {
        'type': 'scipy',
        'maxfun': 200
    },
    'pqc': {
        'type': 'qaoa',
        'layers': 5,
    },
}

TESTER_CONFIG = {
    'pqc': {
        'type': 'wfqaoa',
        'layers': 5,
    }
}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('-w', type=int)
    parser.add_argument('--weights', metavar='N', type=int, nargs='+')
    parser.add_argument('--values', metavar='N', type=int, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    if len(args.weights) != len(args.values):
        raise Exception("Number of provided weights and values must be the same")
    knapsack = KnapsackProblem(max_weight=args.w, items=list(zip(args.weights, args.values)))
   
    qaoa = VQA(knapsack, config=QAOA_CONFIG)
    wfqaoa = VQA(knapsack, config=WFQAOA_CONFIG)
    hqaoa = VQA(knapsack, config=HQAOA_CONFIG)
    tester = VQA(knapsack, config=TESTER_CONFIG)

    output_file = f'{args.w}_{"_".join([f"{w}-{v}" for w,v in list(zip(args.weights, args.values))])}.csv'

    with open(output_file, mode='w') as output:
        print(knapsack.knapsack.items, file=output)
        print("QAOA;WFQAOA;HQAOA", file=output)

        angles = 2 * np.pi * np.random.rand(args.samples, 2, 5)
        weights = 10*np.random.rand(args.samples, 2) + 1
        for i in range(args.samples):
            params_cofing = {
                'angles': angles[i],
                'hyper_args': list(weights[i]) + [weights[i][1]]
            }
            for solver in [qaoa, wfqaoa, hqaoa]:
                best_params = solver.solve(params_cofing)

                print(f"{tester.evaluate(best_params):.3f}", end=';', file=output)
            print(file=output)
