# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


import numpy as np

from typing import Any
from dataclasses import dataclass

from QHyper.converter import Converter
from QHyper.devices.dwave import DWaveDevice
from QHyper.problems import Problem
from QHyper.solvers import Solver, SolverResult


@dataclass
class DQM(Solver):
    """
    DQM solver class.

    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    device : DWaveDevice
        Configuration of the device the solver runs the problem on.
    time : float
        Maximum run time in seconds
    cases: int, default 1
        Number of variable cases (values)
        1 is denoting binary variable.
    """

    problem: Problem
    device: DWaveDevice
    time: float
    cases: int = 1

    def __post_init__(self) -> None:
        self.sampler = self.device.make_sampler('dqm')

    @classmethod
    def from_config(cls, problem: Problem, config: dict[str, Any]) -> 'DQM':
        config = dict(config)
        if 'device' in config:
            config['device'] = DWaveDevice.from_config(config['device'])
        return cls(problem, **config)

    def solve(self) -> SolverResult:
        dqm = Converter.to_dqm(self.problem, self.cases)
        solutions = self.sampler.sample_dqm(dqm, self.time)

        recarray = np.recarray(
            (len(solutions),),
            dtype=([(v, int) for v in solutions.variables]
                   + [('probability', float)]
                   + [('energy', float)])
        )

        num_of_shots = solutions.record.num_occurrences.sum()
        for i, solution in enumerate(solutions.data()):
            for var in solutions.variables:
                recarray[var][i] = solution.sample[var]

            recarray['probability'][i] = (
                solution.num_occurrences / num_of_shots)
            recarray['energy'][i] = solution.energy

        return SolverResult(recarray, {}, [])
