from typing import Union

VARIABLES = Union[tuple[()], tuple[str], tuple[str, str], tuple[str, ...]]
QUBO = dict[VARIABLES, float]
