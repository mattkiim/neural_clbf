import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # or another suitable backend

import glob
import os

import matplotlib.pyplot as plt
from neural_clbf.controllers import NeuralCBFController  # or NeuralCLBFController
from neural_clbf.experiments import (
    ExperimentSuite,
    CBFContourExperiment,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutSuccessRateExperiment,
)

# ------------------------
# 1) Define a single start state for rollouts
#    For Air3D, the state is (x, y, psi).
# ------------------------
start_x = torch.tensor([[0.5, -0.4, 0.0]])  # shape [1, 3]

# ------------------------
# 2) Define nominal params & scenarios for Air3D
# ------------------------
nominal_params = {
    "collisionR": 0.25,
    "velocity": 0.6,
    "omega_max": 1.1,
    "angle_alpha_factor": 1.2,
}
scenarios = [nominal_params]

def plot_air3d():
    # ------------------------
    # 3) Checkpoint directory / file selection
    # ------------------------
    checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/air3d/commit_c69834e/version_0/checkpoints/" 
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    log_file = max(ckpt_files, key=os.path.getctime) if ckpt_files else None
    if log_file is None:
        print("No .ckpt file found in the checkpoint directory!")
        return

    # ------------------------
    # 4) Load the trained neural controller
    # ------------------------
    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)
    # If you trained a NeuralCLBFController instead, import and load that:
    # neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)

    # ------------------------
    # 5) Example: Evaluate V and its gradient over a test batch
    #    We'll sample random states in (x, y, psi).
    # ------------------------
    N_test = 100
    # x ~ U(-1,1), y ~ U(-1,1), psi ~ U(-pi, pi)
    x_vals   = 2.0 * (torch.rand(N_test) - 0.5)  # range (-1,1)
    y_vals   = 2.0 * (torch.rand(N_test) - 0.5)
    psi_vals = math.pi * (2.0 * torch.rand(N_test) - 1.0)  # range (-pi, pi)

    # Stack into shape [N_test, 3]
    x_batch = torch.stack([x_vals, y_vals, psi_vals], dim=1)

    # Evaluate V and its Jacobian
    V, JV = neural_controller.V_with_jacobian(x_batch)
    # JV has shape [N_test, 1, 3]. Squeeze out the '1' dimension
    JV = JV.squeeze(1)  # shape: [N_test, 3]

    # Compute mean gradient over the batch
    mean_gradients = JV.mean(dim=0)
    print("Mean gradient of V wrt state:", mean_gradients)

    # ------------------------
    # 6) Define experiments
    # ------------------------

    # CBF contour experiment: visualize h(x) = boundary_fn(x) in x-y plane
    h_contour_experiment = CBFContourExperiment(
        "h_Contour",
        domain=[(-1.0, 1.0), (-1.0, 1.0)],  # range for x, y
        n_grid=25,
        x_axis_index=0,
        y_axis_index=1,
        x_axis_label="$x$",
        y_axis_label="$y$",
        plot_safe_region=False,
    )
    
    # Rollout success rate experiment: how often do we remain safe, etc.
    # e.g., if you want to test multiple initial states
    # start_xs = torch.tensor([
    #     [0.5, -0.4, 0.0],
    #     [1.0,  0.2, 1.0],
    #     [0.0,  0.8, -1.5],
    # ])
    # rollout_success_experiment = RolloutSuccessRateExperiment(
    #     "Rollout Success",
    #     "CBF Controller",
    #     n_sims=start_xs.shape[0],
    #     t_sim=2.0,
    #     initial_states=start_xs,
    # )

    # ------------------------
    # 7) Create the experiment suite
    #    (choose which experiments to run)
    # ------------------------
    experiment_suite = ExperimentSuite([
        # rollout_experiment,
        h_contour_experiment,
        # V_contour_experiment,
        # rollout_success_experiment,
    ])

    # Associate experiments with the controller
    neural_controller.experiment_suite = experiment_suite

    # ------------------------
    # 8) Run the experiments and plot
    # ------------------------
    neural_controller.experiment_suite.run_all_and_plot(
        neural_controller, display_plots=True
    )

if __name__ == "__main__":
    import math
    plot_air3d()
