import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from neural_clbf.controllers import NeuralCLBFController, NeuralCBFController
from neural_clbf.systems.mvc import MultiVehicleCollision
from neural_clbf.systems.utils import Scenario, lqr, ScenarioList



import matplotlib.pyplot as plt
from neural_clbf.experiments import (
    ExperimentSuite,
    CBFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutSuccessRateExperiment
)

# Create a nominal parameter instance (mock example)
nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
scenarios = [
    nominal_params, # add more for robustness
]

# Create the MultiVehicleCollision instance
mvc = MultiVehicleCollision(
    nominal_params=nominal_params,
    dt=0.01,  # Time step
    controller_dt=None,  # Optional
    scenarios=None,  # Optional
)

# Define the starting state
start_x = torch.tensor([0.0080, -0.5999,  0.5293, -0.2825, -0.4753, -0.3662,  1.5841,  2.6514, 0.6565])

# Plot the rollout
mvc.plot_rollout(start_x)