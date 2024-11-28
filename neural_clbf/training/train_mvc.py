from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCBFController, NeuralObsBFController, NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.systems import MultiVehicleCollision  # Import the new system
from neural_clbf.experiments import (
    ExperimentSuite,
    CBFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutSuccessRateExperiment
)
from neural_clbf.training.utils import current_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05

# Define starting points for simulations for MultiVehicleCollision
start_x = torch.tensor(
    [
        # [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.107, 0.0, -1.107],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, np.pi / 4, -np.pi / 4],
        [0.1, 0.1, -0.1, -0.1, 0.2, -0.2, np.pi / 2, np.pi / 3, -np.pi / 3],
        [-0.2, 0.2, 0.1, -0.1, -0.2, 0.2, np.pi / 4, np.pi / 6, -np.pi / 6],
    ]
)
print(f"start_x dtype: {start_x.dtype}")
print(f"start_x shape: {start_x.shape}")
# quit()
simulation_dt = 0.01

def main(args):
    # Define the scenarios
    nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
    scenarios = [
        nominal_params, # add more for robustness
    ]

    # Initialize the dynamics model with MultiVehicleCollision
    dynamics_model = MultiVehicleCollision()

    # print("\n\ndynamics_model.K:\n")
    # print(dynamics_model.K)

    # Initialize the DataModule with appropriate initial conditions for MultiVehicleCollision
    initial_conditions = [
        (-2, 2),  # x positions
        (-2, 2),  # y positions
        (-2, 2),  # x2 positions
        (-2, 2),  # y2 positions
        (-2, 2),  # x3 positions
        (-2, 2),  # y3 positions
        (-np.pi, np.pi),  # angle 1
        (-np.pi, np.pi),  # angle 2
        (-np.pi, np.pi),  # angle 3
    ]
    
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=2000,
        max_points=20000,
        val_split=0.1,
        batch_size=batch_size,
    )


    # Define the experiment suite
    h_contour_experiment = CBFContourExperiment(
        "h_Contour",
        domain=[(-1.0, 1.0), (-1.0, 1.0)],  # Adjust the range if needed
        n_grid=50,
        x_axis_index=0,  # Index for x position of vehicle 1
        y_axis_index=1,  # Index for y position of vehicle 1
        x_axis_label="$x_1$",
        y_axis_label="$y_1$",
        plot_safe_region=True,  # Set to True if you want to visualize unsafe regions
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        0,  # Index for x position of vehicle 1
        "$x_1$",
        1,  # Index for y position of vehicle 1
        "$y_1$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=1.0,
    )

    rollout_success_experiment = RolloutSuccessRateExperiment(
        "Rollout Success",
        "CLBF Controller",
        n_sims=500,
        t_sim=1.0,
    )

    experiment_suite = ExperimentSuite([h_contour_experiment, rollout_experiment, rollout_success_experiment])
    # experiment_suite = ExperimentSuite([rollout_experiment])


    cbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=2,
        cbf_hidden_size=64,
        cbf_lambda=1.0,
        cbf_relaxation_penalty=50.0, # ?
        controller_period=controller_period,
        primal_learning_rate=1e-3,
        scale_parameter=10.0, # ?
        learn_shape_epochs=0,
        use_relu=False,
    )

    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
        barrier=False,
        disable_gurobi=True,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/multivehicle_collision",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=51,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--clbf_hidden_layers", type=int, default=2, help="Number of hidden layers for CLBF")
    parser.add_argument("--clbf_hidden_size", type=int, default=64, help="Size of hidden layers for CLBF")
    parser.add_argument("--clf_lambda", type=float, default=1.0, help="CLF lambda parameter")
    parser.add_argument("--clf_relaxation_penalty", type=float, default=100.0, help="CLF relaxation penalty")

    # Add PyTorch Lightning trainer arguments
    args = parser.parse_args()

    main(args)
