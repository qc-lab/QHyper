
import sys
sys.path.append("../Qhyper")
import dimod
from QHyper.problems.community_detection import KarateClubNetwork
from QHyper.solvers.converter import Converter
from QHyper.problems.community_detection import \
    CommunityDetectionProblem

dqm = dimod.DiscreteQuadraticModel()

vertices = ["A", "B"]

edges = [("A", "B")]

communities = [0, 1]


dqm = dimod.DiscreteQuadraticModel()

for p in vertices:

    _ = dqm.add_variable(2, label=p)

for p0, p1 in edges:

    dqm.set_quadratic(p0, p1, {(c, c): 1 for c in communities})

#(linear, quadratic, offset)=dqm.to_ising()
# print(linear)
# print("---------")
# print(quadratic)
# print("--------")
# print(offset)
print(dqm.to_numpy_vectors())

#vectors = dqm.to_numpy_vectors()


# new = dimod.DiscreteQuadraticModel.from_numpy_vectors(*vectors)
# RESOLUTION = 0.5

# #problem = KarateClubNetwork(resolution=RESOLUTION)
# problem =CommunityDetectionProblem(
#             network_data=KarateClubNetwork(resolution=RESOLUTION),
#             communities=1
#         )

# #print(problem.objective_function)
# #print("-------------------------")
# #print(problem.constraints)
# qubo = Converter.create_qubo(problem, weights=[1])
# #print(qubo)
# (linear, quadratic, offset)=dimod.BinaryQuadraticModel.from_qubo(qubo).to_ising()
# print(linear)
# print("---------")
# print(quadratic)
# print("--------")
# print(offset)

#print(dimod.BinaryQuadraticModel.from_qubo(qubo).to_ising().quadratic)


