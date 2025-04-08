from argparse import ArgumentParser

import math
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
# Import your new Air3D system class
from neural_clbf.systems.air3d import Air3D  
from neural_clbf.experiments import (
    ExperimentSuite,
    # Potentially you can add experiments: CBFContourExperiment, etc.
    # For now we can leave it empty or add relevant experiments
)
from neural_clbf.training.utils import current_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

def main(args):
    # Define the scenario for Air3D
    nominal_params = {
        "collisionR": 0.25,
        "velocity": 0.6,
        "omega_max": 1.1,
        "angle_alpha_factor": 1.2
    }
    scenarios = [
        nominal_params,  # You can add more scenarios for robustness if desired
    ]

    # Instantiate the Air3D system
    dynamics_model = Air3D(
        nominal_params,
        dt=0.01,
        controller_dt=controller_period,
        scenarios=scenarios
    )

    # Define the initial conditions for (x, y, psi)
    # Each tuple is (min, max). Note that the angle is scaled by angle_alpha_factor
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # y
        (-math.pi * 1.2, math.pi * 1.2),  # psi
    ]

    # Create the episodic data module
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,  # for random sampling only
        trajectory_length=500,
        fixed_samples=10000,
        max_points=10000,
        val_split=0.1,
        batch_size=batch_size,
        quotas={"unsafe": 0.7},  # Attempt to sample more from unsafe regions
    )

    # Define (optional) experiments to run during training
    # For a single vehicle, you may tailor these for visualization
    experiment_suite = ExperimentSuite([])

    # Create the neural CBF controller
    cbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=3,
        cbf_hidden_size=512,
        cbf_lambda=0.0,
        cbf_relaxation_penalty=2e3, 
        controller_period=controller_period,
        primal_learning_rate=1e-3,
        scale_parameter=10.0, 
        learn_shape_epochs=11,
        use_relu=True,
    )

    # Initialize logging
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/air3d",
        name=f"commit_{current_git_hash()}",
    )

    # Create the PyTorch Lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=51,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(cbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--clbf_hidden_layers", type=int, default=2, help="Number of hidden layers for CLBF")
    parser.add_argument("--clbf_hidden_size", type=int, default=64, help="Size of hidden layers for CLBF")
    parser.add_argument("--clf_lambda", type=float, default=1.0, help="CLF lambda parameter")
    parser.add_argument("--clf_relaxation_penalty", type=float, default=100.0, help="CLF relaxation penalty")

    # Parse arguments and run
    args = parser.parse_args()
    main(args)
