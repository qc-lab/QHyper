from abc import abstractmethod
import numpy as np

import numpy.typing as npt
from typing import Callable


class Optimizer:
    @abstractmethod
    def minimize(
        self,
        func: Callable[[npt.NDArray[np.float64]], float],
        init: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        ...
