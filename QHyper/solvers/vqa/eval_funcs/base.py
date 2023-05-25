from abc import abstractmethod

from QHyper.problems.base import Problem


class EvalFunc:
    """
    Abstract base class for evaluation functions.

    Methods
    -------
    evaluate(results, problem, const_params)
        Evaluate the results based on the problem and constant parameters.
    """

    @abstractmethod
    def evaluate(
        self,
        results: dict[str, float],
        problem: Problem,
        const_params: list[float]
    ) -> float:
        """
        Evaluate the results based on the problem and constant parameters.

        Parameters
        ----------
        results : dict[str, float]
            The results to be evaluated.
        problem : Problem
            The problem definition.
        const_params : list[float]
            The constant parameters.

        Returns
        -------
        float
            The evaluation result.
        """
        ...
