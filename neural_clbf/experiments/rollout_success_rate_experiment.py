"""An experiment for testing success rates"""
from typing import List, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import tqdm
import numpy as np

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
        initial_states = None,
        relative = True
    ):
        """Initialize an experiment for simulating controller performance.

        args:
            name: the name of this experiment
            algorithm_name: the name of the algorithm for saving results
            n_sims: the number of random simulations to run
            t_sim: the amount of time to simulate for
        """
        super(RolloutSuccessRateExperiment, self).__init__(name)

        # Save parameters
        self.algorithm_name = algorithm_name
        self.n_sims = n_sims
        self.t_sim = t_sim
        self.initial_states = initial_states
        self.relative = relative

    @torch.no_grad()
    def run2(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment, likely by evaluating the controller, but the experiment
        has freedom to call other functions of the controller as necessary (if these
        functions are not supported by all controllers, then experiments will be
        responsible for checking compatibility with the provided controller)

        args:
            controller_under_test: the controller with which to run the experiment
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # We want to track these metrics, averaged over all simulations
        num_collisions = 0
        num_goals_reached = 0
        total_time_to_goal = 0.0

        # Back up the original scene
        if hasattr(controller_under_test.dynamics_model, "scene"):
            original_scene = controller_under_test.dynamics_model.scene  # type: ignore

        prog_bar_range = tqdm.trange(
            0, self.n_sims, desc="Computing Success Rate...", leave=True
        )
        for sim_idx in prog_bar_range:
            # Generate a random environment
            if hasattr(controller_under_test.dynamics_model, "scene"):
                room_size = 10.0
                num_obstacles = 8
                box_size_range = (0.75, 1.75)
                position_range = (-4.0, 4.0)
                rotation_range = (-np.pi, np.pi)
                scene = Scene([])
                scene.add_walls(room_size)
                scene.add_random_boxes(
                    num_obstacles,
                    box_size_range,
                    position_range,
                    position_range,
                    rotation_range,
                )
                controller_under_test.dynamics_model.scene = scene  # type: ignore

            # Generate a safe starting state
            if self.initial_states is not None:
                # TODO: take in the initial states numpy file
                pass
            x = controller_under_test.dynamics_model.sample_safe(1)
            while controller_under_test.dynamics_model.unsafe_mask(x).any():
                x = controller_under_test.dynamics_model.sample_safe(1)

            # Reset the controller if necessary
            if hasattr(controller_under_test, "reset_controller"):
                controller_under_test.reset_controller(x)  # type: ignore

            # Simulate forward. Stop if we reach the goal or hit something
            dt = controller_under_test.dynamics_model.dt
            controller_update_freq = int(controller_under_test.controller_period / dt)
            num_timesteps = int(self.t_sim // dt)
            for tstep in range(num_timesteps):
                # Get the control input at the current state if it's time
                if tstep % controller_update_freq == 0:
                    u_current = controller_under_test.u(x)

                # Simulate forward using the dynamics
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(
                    x,
                    u_current,
                )
                x = x + dt * xdot

                # Check if we're safe or not
                if controller_under_test.dynamics_model.failure(x).any():
                    num_collisions += 1
                    break

                # Check if we're at the goal or not
                if controller_under_test.dynamics_model.goal_mask(x).any():
                    num_goals_reached += 1
                    total_time_to_goal += tstep * dt
                    break

        # Create a dataframe for storing the results
        results_df = pd.DataFrame(
            [
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "Goal-reaching rate",
                    "Value": num_goals_reached / self.n_sims,
                },
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "Safety rate",
                    "Value": 1 - num_collisions / self.n_sims,
                }, # TODO: implement false positives and false negatives: if value
                {
                    "Algorithm": self.algorithm_name,
                    "Metric": "Time to goal",
                    "Value": total_time_to_goal / num_goals_reached if num_goals_reached != 0 else -1, # TODO: implement concept of goal?
                },
            ]
        )

        # Restore the original scene
        if hasattr(controller_under_test.dynamics_model, "scene"):
            controller_under_test.dynamics_model.scene = original_scene  # type: ignore

        return results_df

    def run3(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment to compute false positives and false negatives based on the CBF.

        args:
            controller_under_test: the controller with which to run the experiment.
        returns:
            a pandas DataFrame containing the results of the experiment.
        """
        # Metrics
        num_false_positives = 0
        num_false_negatives = 0
        num_true_positives = 0
        num_true_negatives = 0
        num_safe_rollouts = 0
        num_unsafe_rollouts = 0

        # Simulation parameters
        dt = controller_under_test.dynamics_model.dt
        controller_update_freq = int(controller_under_test.controller_period / dt)
        num_timesteps = int(self.t_sim // dt)

        if self.initial_states is not None:
            # TODO: take in the initial states numpy file
            initial_states = self.initial_states
            pass
        
        # False Positive Check: Initialize in safe states
        for rollout_idx in range(self.n_sims):
            # Initialize a safe state
            if self.initial_states is not None:
                x = initial_states[rollout_idx].view(1, 9) # TODO: hardcoded 
                # print(x.shape)
            else:
                x = controller_under_test.dynamics_model.sample_safe(1)
                while not controller_under_test.dynamics_model.safe_mask(x).all():
                    x = controller_under_test.dynamics_model.sample_safe(1)
            if self.relative:
                x = controller_under_test.dynamics_model.states_rel(x)
            # Reset the controller if necessary
            if hasattr(controller_under_test, "reset_controller"):
                controller_under_test.reset_controller(x)

            # Simulate forward
            safe_trajectory = True
            for tstep in range(num_timesteps):
                # Get control input if it's time
                if tstep % controller_update_freq == 0:
                    u_current = controller_under_test.u(x)

                # Simulate dynamics
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(x, u_current)
                x = x + dt * xdot

                # Check if the state becomes unsafe
                if controller_under_test.dynamics_model.unsafe_mask(x).any():
                    safe_trajectory = False
                    break

            # Accumulate metrics
            num_safe_rollouts += 1
            if not safe_trajectory:
                num_false_positives += 1
            else:
                num_true_positives += 1

        # False Negative Check: Initialize in unsafe states
        for rollout_idx in range(self.n_sims):
            # Initialize an unsafe state
            if self.initial_states is not None:
                x = initial_states[rollout_idx].view(1, 9) # TODO: hardcoded 
                # print(x.shape)
            else:
                x = controller_under_test.dynamics_model.sample_unsafe(1)
                while not controller_under_test.dynamics_model.unsafe_mask(x).all():
                    x = controller_under_test.dynamics_model.sample_unsafe(1)

            if self.relative:
                x = controller_under_test.dynamics_model.states_rel(x)

            # Reset the controller if necessary
            if hasattr(controller_under_test, "reset_controller"):
                controller_under_test.reset_controller(x)

            # Simulate forward
            safe_trajectory = True
            for tstep in range(num_timesteps):
                # Get control input if it's time
                if tstep % controller_update_freq == 0:
                    u_current = controller_under_test.u(x)

                # Simulate dynamics
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(x, u_current)
                x = x + dt * xdot

                # Check if the state remains safe
                if controller_under_test.dynamics_model.safe_mask(x).all():
                    safe_trajectory = False
                    break

            # Accumulate metrics
            num_unsafe_rollouts += 1
            if safe_trajectory:
                num_false_negatives += 1
            else: 
                num_true_negatives += 1

        # Compute rates
        false_positive_rate = num_false_positives / num_safe_rollouts if num_safe_rollouts > 0 else 0
        false_negative_rate = num_false_negatives / num_unsafe_rollouts if num_unsafe_rollouts > 0 else 0

        false_positive_rate_cm = num_false_positives / (num_false_positives + num_true_negatives)
        false_negative_rate_cm = num_false_negatives / (num_false_negatives + num_true_positives)

        print("FP: ", false_positive_rate)
        print("FN: ", false_negative_rate)

        print("FP CM: ", false_positive_rate_cm)
        print("FN CM: ", false_negative_rate_cm)


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

    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment to compute false positives and false negatives based on the CBF.

        args:
            controller_under_test: the controller with which to run the experiment.
        returns:
            a pandas DataFrame containing the results of the experiment.
        """
        # Metrics
        num_false_positives = 0
        num_false_negatives = 0
        num_true_positives = 0
        num_true_negatives = 0
        num_safe_rollouts = 0
        num_unsafe_rollouts = 0

        # Simulation parameters
        dt = controller_under_test.dynamics_model.dt
        controller_update_freq = int(controller_under_test.controller_period / dt)
        num_timesteps = int(self.t_sim // dt)

        if self.initial_states is not None:
            # TODO: take in the initial states numpy file
            initial_states = self.initial_states
            pass
        
        # False Positive Check: Initialize in safe states
        for rollout_idx in range(self.n_sims):
            # Initialize a safe state
            x = initial_states[rollout_idx].view(1, 9) # TODO: hardcoded 
            if self.relative:
                x = controller_under_test.dynamics_model.states_rel(x)
            # Reset the controller if necessary
            if hasattr(controller_under_test, "reset_controller"):
                controller_under_test.reset_controller(x)

            initially_safe = False if controller_under_test.V(x) < 0 else True

            # Simulate forward
            safe_trajectory = True
            for tstep in range(num_timesteps):
                # Get control input if it's time
                if tstep % controller_update_freq == 0:
                    u_current = controller_under_test.u(x)

                # Simulate dynamics
                xdot = controller_under_test.dynamics_model.closed_loop_dynamics(x, u_current)
                x = x + dt * xdot

                # Check if the state becomes unsafe
                if controller_under_test.dynamics_model.unsafe_mask(x).any():
                    safe_trajectory = False
                    break

            # Accumulate metrics
            if initially_safe:
                num_safe_rollouts += 1
                if safe_trajectory:
                    num_true_positives += 1
                else:
                    num_false_positives += 1
            else:
                num_unsafe_rollouts += 1
                if not safe_trajectory: 
                    num_true_negatives += 1
                else:
                    num_false_negatives += 1

        # Compute rates
        print("TP FP TN FN: ", num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)

        false_positive_rate = num_false_positives / 500
        false_negative_rate = num_false_negatives / 500

        false_positive_rate_cm = num_false_positives / (num_false_positives + num_true_negatives)
        false_negative_rate_cm = num_false_negatives / (num_false_negatives + num_true_positives)

        print("FP: ", false_positive_rate)
        print("FN: ", false_negative_rate)

        print("FP CM: ", false_positive_rate_cm)
        print("FN CM: ", false_negative_rate_cm)


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
            plt.savefig("plot_success_rate.png")
            return []
        else:
            return [fig_handle]
