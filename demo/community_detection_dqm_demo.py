from QHyper.problems.community_detection import BrainNetwork, CommunityDetectionProblem
from QHyper.problems.network_communities.utils import draw_communities
from QHyper.solvers.dqm.dqm import DQM


path = "QHyper/problems/network_communities/brain_community_data"
data_name = "Edge_AAL90_Binary"

name = "brain"
folder = "demo/demo_output"
solution_file = f"{folder}/{name}_dqm_solution.csv"
decoded_solution_file = f"{folder}/{name}_dqm_decoded_solution.csv"
img_solution_path = f"{folder}/{name}_dqm.png"

brain_network = BrainNetwork(input_data_dir=path, input_data_name=data_name)
brain_problem = CommunityDetectionProblem(brain_network, N_communities=3)
problem = brain_problem

dqm = DQM(problem, time=5)
sampleset = dqm.solve()

sample = sampleset.first.sample
print(sample)

draw_communities(
    problem=problem, sample=sample, path=img_solution_path
)
