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


start_x = torch.tensor(
    [
        # [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.107, 0.0, -1.107],
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 4, -np.pi / 4],
        # [0.1, 0.1, 0.4, 0.1, 0.2, 0.4, 0, 0, 0],
        # [-0.2, 0.2, 0.1, -0.1, -0.2, 0.3, np.pi / 4, np.pi / 6, -np.pi / 6],
        [0.0080, -0.5999,  0.5293, -0.2825, -0.4753, -0.3662,  1.5841,  2.6514, 0.6565],
        # [0.0080, -0.5999,  0.5293, -0.2825, -0.4753, -0.3662,  1.5, 1.5, 1.5]
        # [-0.3498,  0.0176,  0.0857, -0.3451,  0.6371,  0.3958, -0.0235, 1.6853, -2.5857]
    ]
)

file_path = 'initial_conditions.npy'
initial_conditions = np.load(file_path)

start_xs = torch.tensor(initial_conditions[:, :-1], dtype=torch.float32)

nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
scenarios = [
    nominal_params, # add more for robustness
]


def plot_mvc_rel():
    # log_file = "../training/logs/multivehicle_collision/commit_97da873/clbf_1/checkpoints/epoch=50-step=1478.ckpt"
    # neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    log_file = "../training/logs/multivehicle_collision/commit_0af93dc/version_14/checkpoints/epoch=100-step=3029.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # log_file = "../training/logs/multivehicle_collision/commit_0af93dc/version_17/checkpoints/epoch=100-step=3029.ckpt"
    # neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # log_file = "../training/logs/multivehicle_collision/commit_0af93dc/version_20/checkpoints/epoch=100-step=3029.ckpt"
    # neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # log_file = "../training/logs/multivehicle_collision/commit_0af93dc/version_22/checkpoints/epoch=100-step=5857.ckpt"
    # neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # log_file = "../training/logs/multivehicle_collision/commit_0af93dc/version_25/checkpoints/epoch=200-step=3617.ckpt"
    # neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # log_file = "../training/logs/multivehicle_collision/commit_0af93dc/version_26/checkpoints/epoch=300-step=5417.ckpt"
    # neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    log_file = "/home/mk0617/Documents/dev/research/SASLab/neural_clbf/neural_clbf/training/logs/multivehicle_collision/commit_c3947d8/version_0/checkpoints/epoch=101-step=17951.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)

    # log_file = "/home/mk0617/Documents/dev/research/SASLab/neural_clbf/neural_clbf/training/logs/multivehicle_collision/commit_c3947d8/version_1/checkpoints/epoch=101-step=17951.ckpt"
    # neural_controller = NeuralCBFController.load_from_checkpoint(log_file)



    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        0,  # Index for x position of vehicle 1
        "$x_1$",
        1,  # Index for y position of vehicle 1
        "$y_1$",
        other_index=[2, 3, 4, 5],  # Indices for x2, y2, x3, y3
        other_label=["$x_2$", "$y_2$", "$x_3$", "$y_3$"],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=1.0,
    )

    h_contour_experiment = CBFContourExperiment(
        "h_Contour",
        domain=[(-1.0, 1.0), (-1.0, 1.0)],  # Adjust the range if needed
        n_grid=50,
        x_axis_index=0,  # Index for x position of vehicle 1
        y_axis_index=1,  # Index for y position of vehicle 1
        x_axis_label="$x$",
        y_axis_label="$y$",
        plot_safe_region=False,  # Set to True if you want to visualize unsafe regions
    )

    rollout_success_experiment = RolloutSuccessRateExperiment(
        "Rollout Success",
        "CBF Controller",
        n_sims=500,
        t_sim=1.0,
        initial_states=start_xs
    )

    experiment_suite = ExperimentSuite([rollout_experiment, h_contour_experiment])
    # experiment_suite = ExperimentSuite([rollout_experiment, rollout_success_experiment])
    # experiment_suite = ExperimentSuite([h_contour_experiment])
    # experiment_suite = ExperimentSuite([rollout_success_experiment])
    # experiment_suite = ExperimentSuite([rollout_experiment])



    neural_controller.experiment_suite = experiment_suite

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    plot_mvc_rel()
