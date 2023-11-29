
import sys
sys.path.append("../Qhyper")
import dimod
from QHyper.problems.community_detection import KarateClubNetwork
from QHyper.solvers.converter import Converter
from QHyper.problems.community_detection import \
    CommunityDetectionProblem
RESOLUTION = 0.5

#problem = KarateClubNetwork(resolution=RESOLUTION)
problem =CommunityDetectionProblem(
            network_data=KarateClubNetwork(resolution=RESOLUTION),
            communities=1
        )

print(problem.objective_function)
print("-------------------------")
#print(problem.constraints)
qubo = Converter.create_qubo(problem, weights=[1])
#print(qubo)
(linear, quadratic, offset)=dimod.BinaryQuadraticModel.from_qubo(qubo).to_ising()
print(linear)
print("---------")
print(quadratic)
print("--------")
print(offset)

#print(dimod.BinaryQuadraticModel.from_qubo(qubo).to_ising().quadratic)


