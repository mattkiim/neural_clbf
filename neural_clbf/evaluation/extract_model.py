import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from neural_clbf.controllers import NeuralCLBFController, NeuralCBFController


import matplotlib.pyplot as plt
from neural_clbf.experiments import (
    ExperimentSuite,
    CBFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutSuccessRateExperiment
)


file_path = 'initial_conditions_2.npy'
initial_conditions = np.load(file_path)

start_xs = torch.tensor(initial_conditions[:, :-1], dtype=torch.float32)

nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
scenarios = [
    nominal_params, # add more for robustness
]


def plot_mvc_rel():
    log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_3a414cf/version_71/checkpoints/epoch=100-step=372731.ckpt" # combined_states
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    print()
    torch.save(neural_controller.V_nn.state_dict(), 'mvc9d_neural_cbf.pth')




if __name__ == "__main__":
    plot_mvc_rel()
