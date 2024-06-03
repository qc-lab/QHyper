# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

import importlib
import importlib.util
import pathlib
import os
import inspect
import re

import numpy as np
import numpy.typing as npt

from typing import Callable, NewType

Array1D = NewType("Array1D", npt.NDArray)
Array2D = NewType("Array2D", npt.NDArray)
ArrayND = NewType("ArrayND", npt.NDArray)


def weighted_avg_evaluation(
    results: np.recarray,
    score_function: Callable[[np.record, float], float],
    penalty: float = 0,
    limit_results: int | None = None,
    normalize: bool = True,
) -> float:
    score: float = 0

    sorted_results = sort_solver_results(results, limit_results)
    if normalize:
        scaler = 1 / sorted_results.probability.sum()
    else:
        scaler = 1
    for rec in sorted_results:
        score += scaler * rec.probability * score_function(rec, penalty)
    return score


def sort_solver_results(
    results: np.recarray,
    limit_results: int | None = None,
) -> np.recarray:
    limit_results = limit_results or len(results)
    results_ = np.sort(results, order='probability')
    return results_[::-1][:limit_results]


def add_evaluation_to_results(
    results: np.recarray,
    score_function: Callable[[np.record, float], float],
    penalty: float = 0,
) -> np.recarray:
    if 'evaluation' in results.dtype.names:
        return results

    new_dtype = np.dtype(results.dtype.descr + [('evaluation', 'f8')])
    new_results = np.empty(results.shape, dtype=new_dtype)
    new_results['evaluation'] = [score_function(x, penalty) for x in results]

    for dtype in results.dtype.names:
        new_results[dtype] = results[dtype]

    return new_results


def class_to_snake(cls: type) -> str:
    if hasattr(cls, 'name'):
        return cls.name
    if cls.__name__.isupper():
        return cls.__name__.lower()
    return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()


def search_for(class_type: type, path: str) -> dict[str, type]:
    cwd = os.getcwd()
    _path = pathlib.Path(path)
    classes = {}
    if _path.is_file():
        if _path.name.endswith('.py') and _path.name != '__init__.py':
            module_name = _path.name[:-3]
            module_path = os.path.join(cwd, _path)

            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, module_path)
                assert spec is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for _, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, class_type)
                        and obj != class_type
                    ):
                        classes[class_to_snake(obj)] = obj
                        print(f"Imported {obj} from {module_path}")

            except Exception as e:
                print(f"Failed to import {module_name} from {_path}: {e}")

    elif _path.is_dir():
        for item in _path.iterdir():
            classes |= search_for(class_type, str(item))
    return classes
