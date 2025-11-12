# This work was supported by the EuroHPC PL infrastructure funded at the
# Smart Growth Operational Programme (2014-2020), Measure 4.2
# under the grant agreement no. POIR.04.02.00-00-D014/20-00


from typing import Any
import numpy as np
from dataclasses import dataclass
import random

from QHyper.problems.base import Problem
from QHyper.problems.community_detection import CommunityDetectionProblem
from QHyper.solvers.base import Solver, SolverResult

from simanneal import Annealer


class CommunityDetectionAnnealer(Annealer):    
    def __init__(self, problem: CommunityDetectionProblem, initial_state: dict[int, int]):
        self.problem = problem
        self.num_nodes = len(self.problem.community)
        self.num_communities = self.problem.cases
        self.B = self.problem.B 
        super().__init__(initial_state)
    
    def move(self):
        node_id = random.randint(0, self.num_nodes - 1)
        
        current_community = self.state[node_id]
        new_community = random.randint(0, self.num_communities - 1)
        
        while new_community == current_community and self.num_communities > 1:
            new_community = random.randint(0, self.num_communities - 1)
        
        self.state[node_id] = new_community
    
    def energy(self):
        modularity = 0.0
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.state[i] == self.state[j]:
                    modularity += self.B[i, j]
        
        return -modularity


class GenericProblemAnnealer(Annealer):
    def __init__(self, problem: Problem, initial_state: Any):
        self.problem = problem
        super().__init__(initial_state)
    
    def move(self):
        pass
    
    def energy(self):
        return 0.0


@dataclass
class SimulatedAnnealingSolver(Solver):
    """
    Classical simulated Annealing solver.
    
    Attributes
    ----------
    problem : Problem
        The problem to be solved.
    Tmax : float, optional, default=25000.0
        Maximum temperature for annealing schedule.
    Tmin : float, optional, default=2.5
        Minimum temperature for annealing schedule.
    steps : int, optional, default=50000
        Number of iterations to perform.
    updates : int, optional, default=100
        Number of updates to show during annealing (for progress reporting).
    """
    
    problem: Problem
    Tmax: float = 25000.0
    Tmin: float = 2.5
    steps: int = 50000
    updates: int = 100
    
    def __init__(
        self,
        problem: Problem,
        Tmax: float = 25000.0,
        Tmin: float = 2.5,
        steps: int = 50000,
        updates: int = 100,
        **config: Any
    ) -> None:
        self.problem = problem
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.steps = steps
        self.updates = updates
        self.config = config
    
    def _generate_initial_state(self) -> dict[int, int]:
        if isinstance(self.problem, CommunityDetectionProblem):
            num_nodes = len(self.problem.community)
            num_communities = self.problem.cases
            return {i: random.randint(0, num_communities - 1) for i in range(num_nodes)}
        else:
            return {}
    
    def _create_annealer(self, initial_state: Any) -> Annealer:
        if isinstance(self.problem, CommunityDetectionProblem):
            return CommunityDetectionAnnealer(self.problem, initial_state)
        else:
            return GenericProblemAnnealer(self.problem, initial_state)
    
    def _state_to_solution(self, state: dict[int, int]) -> dict[str, int]:
        solution = {}
        
        if isinstance(self.problem, CommunityDetectionProblem):
            if self.problem.one_hot_encoding:
                for node_id, community_id in state.items():
                    for case_val in range(self.problem.cases):
                        var_id = node_id * self.problem.cases + case_val
                        var_name = f"s{var_id}"
                        solution[var_name] = 1 if case_val == community_id else 0
            else:
                for node_id, community_id in state.items():
                    var_name = f"x{node_id}"
                    solution[var_name] = community_id
        
        return solution
    
    def solve(self) -> SolverResult:
        initial_state = self._generate_initial_state()
        annealer = self._create_annealer(initial_state)
        
        annealer.Tmax = self.Tmax
        annealer.Tmin = self.Tmin
        annealer.steps = self.steps
        annealer.updates = self.updates
        
        final_state, final_energy = annealer.anneal()
        solution = self._state_to_solution(final_state)
        
        all_vars = list(self.problem.objective_function.get_variables())
        for constraint in self.problem.constraints:
            all_vars.extend(constraint.get_variables())
        all_vars = sorted(set(all_vars))
        
        result = np.recarray(
            (1,),
            dtype=[(str(v), int) for v in all_vars] + [('probability', float), ('energy', float)]
        )
        
        for var in all_vars:
            result[str(var)][0] = solution.get(str(var), 0)
        
        result['probability'][0] = 1.0
        result['energy'][0] = final_energy
        
        return SolverResult(
            result,
            {
                'Tmax': self.Tmax,
                'Tmin': self.Tmin,
                'steps': self.steps,
                'final_energy': final_energy
            },
            []
        )
