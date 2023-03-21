from typing import Any, Callable


class Optimizer:
    def minimize(
        self,
        func: Callable[[list[float]], float],
        init: list[float]
    ) -> tuple[float, list[float]]:
        ...
