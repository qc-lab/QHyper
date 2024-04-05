import multiprocessing as mp
from typing import Callable

import numpy.typing as npt
from tqdm import tqdm

from QHyper.optimizers.base import OptimizationResult


def run_parallel(
        func: Callable[[npt.NDArray], OptimizationResult],
        args: npt.NDArray, processes: int,
        disable_tqdm: bool = True
) -> list[OptimizationResult]:
    if processes == -1:
        return list(tqdm((func(arg) for arg in args),
                         total=len(args), disable=disable_tqdm))

    with mp.Pool(processes=processes) as pool:
        return list(tqdm(
            pool.imap(func, args), total=len(args), disable=disable_tqdm))
