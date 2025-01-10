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

file_path = 'initial_conditions_2.npy'
initial_conditions = np.load(file_path)

start_xs = torch.tensor(initial_conditions[:, :-1], dtype=torch.float32)

nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
scenarios = [
    nominal_params, # add more for robustness
]


def plot_mvc_rel():
    # log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_28/checkpoints/epoch=200-step=113162.ckpt"
    # log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_31/checkpoints/epoch=50-step=71756.ckpt"
    log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_32/checkpoints/epoch=100-step=284112.ckpt"
    # log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_33/checkpoints/epoch=60-step=85826.ckpt"
    # log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_34/checkpoints/epoch=60-step=85826.ckpt"
    # log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_35/checkpoints/epoch=60-step=85826.ckpt"
    # log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_36/checkpoints/epoch=52-step=74570.ckpt"
    log_file = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_f07d8f3/version_37/checkpoints/epoch=51-step=219387.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)



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
        n_grid=25,
        x_axis_index=0,  # Index for x position of vehicle 1
        y_axis_index=1,  # Index for y position of vehicle 1
        x_axis_label="$x$",
        y_axis_label="$y$",
        plot_safe_region=False,  # Set to True if you want to visualize unsafe regions
    )

    rollout_success_experiment = RolloutSuccessRateExperiment(
        "Rollout Success",
        "CBF Controller",
        n_sims=start_xs.shape[0],
        t_sim=1.0,
        initial_states=start_xs
    )

    # experiment_suite = ExperimentSuite([rollout_experiment, h_contour_experiment])
    # experiment_suite = ExperimentSuite([rollout_experiment, rollout_success_experiment])
    # experiment_suite = ExperimentSuite([h_contour_experiment])
    experiment_suite = ExperimentSuite([rollout_success_experiment])
    # experiment_suite = ExperimentSuite([rollout_experiment])



    neural_controller.experiment_suite = experiment_suite

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )


if __name__ == "__main__":
    plot_mvc_rel()
