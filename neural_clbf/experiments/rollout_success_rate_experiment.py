"""An experiment for testing success rates"""
from typing import List, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import time

from neural_clbf.experiments import Experiment
from neural_clbf.systems.planar_lidar_system import Scene

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, NeuralObsBFController  # noqa
    from neural_clbf.systems import ObservableSystem  # noqa


class RolloutSuccessRateExperiment(Experiment):
    """An experiment for plotting rollout performance of controllers.

    Plots trajectories projected onto a 2D plane.
    """

    def __init__(
        self,
        name: str,
        algorithm_name: str,
        n_sims: int = 500,
        t_sim: float = 10.0,
        initial_states=None,
        relative=False,
    ):
        """Initialize an experiment for simulating controller performance.

        args:
            name: the name of this experiment
            algorithm_name: the name of the algorithm for saving results
            n_sims: the number of random simulations to run
            t_sim: the amount of time to simulate for
        """
        super().__init__(name)

        # Save parameters
        self.algorithm_name = algorithm_name
        self.n_sims = n_sims
        self.t_sim = t_sim
        self.initial_states = initial_states
        self.relative = relative

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment to compute false positives and false negatives based on the CBF.

        args:
            controller_under_test: the controller with which to run the experiment.
        returns:
            a pandas DataFrame containing the results of the experiment.
        """
        # Simulation parameters
        dt = controller_under_test.dynamics_model.dt
        controller_update_freq = int(controller_under_test.controller_period / dt)
        num_timesteps = int(self.t_sim // dt)

        # Metrics we will compute
        num_false_positives = 0
        num_false_negatives = 0
        num_true_positives = 0
        num_true_negatives = 0

        # Grab initial states
        # expected shape: [n_sims, state_dim], e.g. [500, 9]
        if self.initial_states is None:
            raise ValueError("No initial states provided to the experiment.")

        # Make sure it's a tensor on the same device as the controller
        device = next(controller_under_test.parameters()).device

        x = self.initial_states.to(device)

        # If needed, transform states into 'relative' form
        if self.relative:
            x = controller_under_test.dynamics_model.states_rel(x)

        # if hasattr(controller_under_test, "reset_controller"):
        #     # If necessary to reset a loop, do it once per trajectory,
        #     # but that kills parallelization. TODO: implement a batched reset.
        #     pass

        # Determine which trajectories are initially "safe" vs "unsafe"
        # We'll define "safe" if V(x) >= 0
        V_init = controller_under_test.V(x)
        initially_safe_mask = V_init <= 0


        # Keep a boolean mask of which trajectories are still safe
        # i.e., haven't gone into the unsafe set yet
        still_safe = torch.ones(self.n_sims, dtype=torch.bool, device=device)

        start = time.time()
        # Batched simulation
        for tstep in range(num_timesteps):
            # Update control if it's a control timestep
            if tstep % controller_update_freq == 0:
                u_current = controller_under_test.u(x)  # shape: [n_sims, u_dim]

            # Step forward
            xdot = controller_under_test.dynamics_model.closed_loop_dynamics(x, u_current)

            x = x + dt * xdot

            # Check which have become unsafe
            unsafe_mask = controller_under_test.dynamics_model.unsafe_mask(x)  # shape [n_sims] 
            still_safe = still_safe & (~unsafe_mask)

            # if tstep == 5: quit()

        print("TIME:", time.time() - start)

        # Now figure out the confusion matrix
        # True Positive (TP): started safe, ended safe
        # False Positive (FP): started safe, ended unsafe
        # True Negative (TN): started unsafe, ended unsafe
        # False Negative (FN): started unsafe, ended safe
        n_safe = initially_safe_mask.sum().item()

        n_unsafe = self.n_sims - n_safe

        still_safe = still_safe.view(-1, 1)  # shape [n_sims, 1]

        # TP: (initially safe) & (still safe)
        num_true_positives = (initially_safe_mask & still_safe).sum().item()
        # print(initially_safe_mask, still_safe)
        # FP: (initially safe) & (became unsafe)
        num_false_positives = n_safe - num_true_positives

        # FN: (initially unsafe) & (ended safe)
        num_false_negatives = ((~initially_safe_mask) & still_safe).sum().item()
        # TN: (initially unsafe) & (ended unsafe)
        num_true_negatives = n_unsafe - num_false_negatives

        # Print debug
        print("TP, FP, TN, FN:", num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)

        # Overall rates out of 500 (or self.n_sims)
        false_positive_rate = num_false_positives / self.n_sims
        false_negative_rate = num_false_negatives / self.n_sims

        # Rates out of their respective confusion matrix rows
        # FP rate = FP / (FP + TN) = FP / ( # initially safe? ) depending on definition
        # but let's keep your original definitions
        if (num_false_positives + num_true_negatives) > 0:
            false_positive_rate_cm = (
                num_false_positives / (num_false_positives + num_true_negatives)
            )
        else:
            false_positive_rate_cm = 0.0

        if (num_false_negatives + num_true_positives) > 0:
            false_negative_rate_cm = (
                num_false_negatives / (num_false_negatives + num_true_positives)
            )
        else:
            false_negative_rate_cm = 0.0

        print("FPR (overall): ", false_positive_rate)
        print("FNR (overall): ", false_negative_rate)
        print("Success Rate: ", 1 - false_negative_rate - false_positive_rate)

        # print("FPR (CM): ", false_positive_rate_cm)
        # print("FNR (CM): ", false_negative_rate_cm)

        # Create a DataFrame with the results
        results_df = pd.DataFrame(
            [
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "False Positive Rate",
                    "Value": false_positive_rate,
                },
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "False Negative Rate",
                    "Value": false_negative_rate,
                },
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "False Positive Rate CM",
                    "Value": false_positive_rate_cm,
                },
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "False Negative Rate CM",
                    "Value": false_negative_rate_cm,
                },
            ]
        )

        return results_df

    def plot(
        self,
        controller_under_test: "Controller",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, figure]]:
        """
        Plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Set up the plot
        fig, ax = plt.subplots()

        sns.barplot(x="Metric", y="Value", hue="Algorithm", ax=ax, data=results_df)

        fig_handle = ("Controller performance", fig)

        if display_plots:
            plt.show()
            # Save the figure only if you truly want to—be mindful that it blocks if plt.show() is used
            plt.savefig("plot_success_rate.png")
            return []
        else:
            return [fig_handle]
