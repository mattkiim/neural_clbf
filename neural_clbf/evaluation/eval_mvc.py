import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import glob
import os


from neural_clbf.controllers import NeuralCLBFController, NeuralCBFController


import matplotlib.pyplot as plt
from neural_clbf.experiments import (
    ExperimentSuite,
    CBFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutSuccessRateExperiment,
)


start_x = torch.tensor(
    [
        [0.0080, -0.5999,  0.5293, -0.2825, -0.4753, -0.3662, 1.5841, 2.6514, 0.6565],
    ]
)

file_path = 'boundary_initials.npy'
# file_path = 'initial_conditions_2.npy'
# file_path = 'initial_states_5000.npy'
# file_path = 'initial_states_all_10000.npy'

initial_conditions = np.load(file_path)

# print(initial_conditions.shape); quit()
start_x = torch.tensor([initial_conditions[3, :-1]])
# print(start_x); quit()

start_xs = torch.tensor(initial_conditions[:, :-1], dtype=torch.float32)

nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
scenarios = [
    nominal_params, # add more for robustness
]


def plot_mvc_rel():
    checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_58/checkpoints/" # gamma=0.5
    # checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_63/checkpoints/" # gamma=1.0
    # checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_65/checkpoints/" # gamma=0.5, sanity check

    checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_0ce2993/version_4/checkpoints/" # gamma=0.5

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    # Select the latest checkpoint file and store it in log_file
    log_file = max(ckpt_files, key=os.path.getctime) if ckpt_files else None

    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)


    # # Load the full PyTorch Lightning checkpoint
    # checkpoint = torch.load(log_file, map_location=torch.device("cpu"))

    # # Extract and save only the model state_dict (weights)
    # torch.save(checkpoint["state_dict"], "model.pth")

    # print(checkpoint["state_dict"])
    # print("Model weights extracted and saved as model.pth")

    # quit()


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
