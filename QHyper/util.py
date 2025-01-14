# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00

"""
This module contains utility functions that are used across the project.

.. rubric:: Functions

.. autofunction:: weighted_avg_evaluation
.. autofunction:: sort_solver_results
.. autofunction:: add_evaluation_to_results
.. autofunction:: search_for

"""


import importlib
import importlib.util
import pathlib
import os
import inspect
import re

import numpy as np
import numpy.typing as npt

from typing import Callable, NewType, Any

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
    """Calculate weighted average evaluation of results.

    Example:

    .. code-block:: python

        results = solver.solve()
        score = weighted_avg_evaluation(
            results.probabilities, solver.problem.get_score, penalty=3,
            limit_results=100, normalize=True)


    Parameters
    ----------
    results : np.recarray
        Results to evaluate. It should contain variables and probability.
    score_function : Callable[[np.record, float], float]
        Function to evaluate results. Most often it's a problem's get_score
        method.
    penalty : float, optional
        Penalty for the constraint violation, by default 0
    limit_results : int, optional
        Number of results to evaluate, by default None
    normalize : bool, optional
        Normalize the score, by default True, applicable when the limit is set

    Returns
    -------
    float
        Weighted average evaluation of results.
    """    

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
    """Sort solver results by probability.

    Example:

    .. code-block:: python

        results = solver.solve()
        sorted_results = sort_solver_results(results.probabilities, 100)
    
    Parameters
    ----------
    results : np.recarray
        Results to sort. It should contain variables and probability.
    limit_results : int, optional
        Number of results to return, by default None
    
    Returns
    -------
    np.recarray
        Sorted results.    
    """

    limit_results = limit_results or len(results)
    results_ = np.sort(results, order='probability')
    return results_[::-1][:limit_results]


def add_evaluation_to_results(
    results: np.recarray,
    score_function: Callable[[np.record, float], float],
    penalty: float = 0,
) -> np.recarray:
    """Add evaluation to results.

    Example:

    .. code-block:: python
    
        results = solver.solve()
        add_evaluation_to_results(
            results.probabilities, solver.problem.get_score)

    Parameters
    ----------
    results : np.recarray
        Results to evaluate. It should contain variables and probability.
    score_function : Callable[[np.record, float], float]
        Function to evaluate results. Most often it's a problem's get_score
        method.
    penalty : float, optional
        Penalty for the constraint violation, by default 0

    Returns
    -------
    np.recarray
        Results with evaluation added. Can be found under 'evaluation' key.
    """

    if 'evaluation' in results.dtype.names:
        return results

    new_dtype = np.dtype(results.dtype.descr + [('evaluation', 'f8')])
    new_results = np.empty(results.shape, dtype=new_dtype)
    new_results['evaluation'] = [score_function(x, penalty) for x in results]

    for dtype in results.dtype.names:
        new_results[dtype] = results[dtype]

    return new_results


def get_class_name(cls: type) -> str:
    if hasattr(cls, 'name'):
        return cls.name
    return cls.__name__


def search_for(class_type: type, path: str) -> dict[str, type]:
    """This function searches for classes of a given type in a given path.

    If class contains a name attribute, it will be used as a key in the
    returned dictionary. Otherwise, the class name will be used.
    Either way, the key will be lowercased.
    
    Parameters
    ----------
    class_type : type
        Type of the class to search for e.g. Problem, Solver.
    path : str
        Path to the file or directory to search in. 

    Returns
    -------
    dict[str, type]
        Dictionary of found classes with their names as keys and classes as 
        values.
    """

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
                        class_name = get_class_name(obj)
                        classes[class_name.lower()] = obj
                        print(f"Imported {obj} from {module_path}"
                              f" as {class_name}")

            except Exception as e:
                print(f"Failed to import {module_name} from {_path}: {e}")

    elif _path.is_dir():
        for item in _path.iterdir():
            classes |= search_for(class_type, str(item))
    return classes
