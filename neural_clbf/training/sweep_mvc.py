from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch
import numpy as np
import pytorch_lightning as pl

from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.systems import MultiVehicleCollision
from neural_clbf.experiments import ExperimentSuite


def train_model(config, checkpoint_dir=None):
    from argparse import Namespace
    args = Namespace(max_epochs=21, gpus=1 if torch.cuda.is_available() else 0)

    scenarios = [{"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}]
    dynamics_model = MultiVehicleCollision(scenarios[0], dt=0.01, controller_dt=0.01, scenarios=scenarios)

    initial_conditions = [(-1, 1)] * 6 + [(-np.pi * 1.2, np.pi * 1.2)] * 3

    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=100,
        trajectory_length=500,
        fixed_samples=50000,
        max_points=50000,
        val_split=0.1,
        batch_size=64,
        quotas={"boundary": 0.3, "safe": 0.2, "unsafe": 0.3},
    )

    experiment_suite = ExperimentSuite([])

    cbf_controller = NeuralCBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=config["cbf_hidden_layers"],
        cbf_hidden_size=config["cbf_hidden_size"],
        cbf_lambda=config["cbf_lambda"],
        cbf_relaxation_penalty=config["cbf_relaxation_penalty"],
        controller_period=0.01,
        primal_learning_rate=config["primal_learning_rate"],
        scale_parameter=config["scale_parameter"],
        learn_shape_epochs=config["learn_shape_epochs"],
        use_relu=config["use_relu"],
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        enable_checkpointing=False,
        gpus=args.gpus,
        enable_progress_bar=False
    )
    trainer.fit(cbf_controller)


# 🔍 Hyperparameter search space
search_space = {
    "cbf_hidden_layers": tune.choice([3]),
    "cbf_hidden_size": tune.choice([512]),
    "cbf_lambda": tune.choice([0.0]),
    "primal_learning_rate": tune.loguniform(1e-5, 1e-3),
    "scale_parameter": tune.choice([1.0, 5.0, 10.0, 20.0]),
    "use_relu": tune.choice([True, False]),
    "cbf_relaxation_penalty": tune.choice([1e2, 1e3, 2e3, 5e3]),
    "learn_shape_epochs": tune.choice([5, 10, 20]),
}

# 🚀 Run the sweep
tune.run(
    train_model,
    config=search_space,
    resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
    num_samples=20,  # increase for more coverage
    scheduler=ASHAScheduler(metric="val_loss", mode="min"),
    local_dir="ray_results",
    name="cbf_sweep_extended"
)
