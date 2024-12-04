import multiprocessing as mp
from typing import Callable

from tqdm import tqdm

from QHyper.optimizers.base import OptimizationResult


def run_parallel(
        func: Callable[[list[float]], OptimizationResult],
        args: list[list[float]],
        processes: int,
        disable_tqdm: bool = True
) -> list[OptimizationResult]:
    if processes == 1:
        results = []
        for arg in tqdm(args, disable=disable_tqdm):
            result = func(arg)
            results.append(result)
        return results

    with mp.Pool(processes=processes) as pool:
        return list(tqdm(
            pool.imap(func, args), total=len(args), disable=disable_tqdm))
