# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

import importlib
import pathlib
import os
import inspect

import numpy.typing as npt

from typing import Callable, NewType

Array1D = NewType("Array1D", npt.NDArray)
Array2D = NewType("Array2D", npt.NDArray)
ArrayND = NewType("ArrayND", npt.NDArray)


def weighted_avg_evaluation(
    results: dict[str, float],
    score_function: Callable[[str, float], float],
    penalty: float = 0,
    limit_results: int | None = None,
    normalize: bool = True,
) -> float:
    score: float = 0

    sorted_results = sort_solver_results(results, limit_results)
    if normalize:
        scaler = 1 / sum([v for v in sorted_results.values()])
    else:
        scaler = 1

    for result, prob in sorted_results.items():
        score += scaler * prob * score_function(result, penalty)
    return score


def sort_solver_results(
    results: dict[str, float],
    limit_results: int | None = None,
) -> dict[str, float]:
    limit_results = limit_results or len(results)
    return {
        k: v for k, v
        in sorted(results.items(), key=lambda item: item[1],
                  reverse=True)[:limit_results]
    }


def add_evaluation_to_results(
    results: dict[str, float],
    score_function: Callable[[str, float], float],
    penalty: float = 0,
) -> dict[str, tuple[float, float]]:
    """
    Parameters
    ----------
    results : dict[str, float]
        dictionary of results
    score_function : Callable[[str, float], float]
        function that receives result and penalty and returns score, probably
        will be passed from Problem.get_score

    Returns
    -------
    dict[str, tuple[float, float]]
        dictionary of results with scores
    """

    return {k: (v, score_function(k, penalty)) for k, v in results.items()}


def search_for(class_type: type, path: str,  depth: int = 0) -> list[type]:
    cwd = os.getcwd()

    _path = pathlib.Path(path)
    print(f"Searching for {class_type} in {_path}")
    solvers = []
    if _path.is_file():
        if _path.name.endswith('.py') and _path.name != '__init__.py':
            module_name = _path.name[:-3]  # Remove the ".py" extension
            module_path = os.path.join(cwd, _path)
            # Try to import the module
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, class_type)
                        and obj != class_type
                    ):
                        solvers.append(obj)
                        print(f"Found solver: {obj}")

            except Exception as e:
                print(f"Failed to import {module_name}: {e}")

            # loc.append(str(_path.resolve()))  # append to list if match
    elif _path.is_dir() and depth < 2:  # check depth
        for item in _path.iterdir():   # iterate directory contents
            print(f"Searching in {item}")
            solvers += search_for(type, str(item), depth + 1)
    return solvers
