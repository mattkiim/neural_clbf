"""Plot a CBF contour"""
from typing import cast, List, Tuple, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm
import numpy as np


from neural_clbf.experiments import Experiment

if TYPE_CHECKING:
    from neural_clbf.controllers import Controller, CBFController  # noqa


class CBFContourExperiment(Experiment):
    """An experiment for plotting the contours of learned CBFs"""

    def __init__(
        self,
        name: str,
        domain: Optional[List[Tuple[float, float]]] = None,
        n_grid: int = 50,
        x_axis_index: int = 0,
        y_axis_index: int = 1,
        x_axis_label: str = "$x$",
        y_axis_label: str = "$y$",
        default_state: Optional[torch.Tensor] = None,
        plot_safe_region: bool = True,
    ):
        """Initialize an experiment for plotting the value of the CBF over selected
        state dimensions.

        args:
            name: the name of this experiment
            domain: a list of two tuples specifying the plotting range,
                    one for each state dimension.
            n_grid: the number of points in each direction at which to compute h
            x_axis_index: the index of the state variable to plot on the x axis
            y_axis_index: the index of the state variable to plot on the y axis
            x_axis_label: the label for the x axis
            y_axis_label: the label for the y axis
            default_state: 1 x dynamics_model.n_dims tensor of default state
                           values. The values at x_axis_index and y_axis_index will be
                           overwritten by the grid values.
            plot_safe_region: True to plot the safe/unsafe region boundaries.
        """
        super(CBFContourExperiment, self).__init__(name)

        # Default to plotting over [-1, 1] in all directions
        if domain is None:
            domain = [(-1.0, 1.0), (-1.0, 1.0)]
        self.domain = domain

        self.n_grid = n_grid
        self.x_axis_index = x_axis_index
        self.y_axis_index = y_axis_index
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.default_state = default_state
        self.plot_safe_region = plot_safe_region

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
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
        # Sanity check: can only be called on a NeuralCBFController
        if not hasattr(controller_under_test, "V"):
            raise ValueError("Controller under test must be a CBFController")

        controller_under_test = cast("CBFController", controller_under_test)

        # Set up a dataframe to store the results
        results = []

        # Set up the plotting grid
        device = "cpu"
        if hasattr(controller_under_test, "device"):
            device = controller_under_test.device  # type: ignore

        x_vals = torch.linspace(
            self.domain[0][0], self.domain[0][1], self.n_grid, device=device
        )
        y_vals = torch.linspace(
            self.domain[1][0], self.domain[1][1], self.n_grid, device=device
        )

        # Default to all zeros if no default provided
        if self.default_state is None:
            default_state = torch.zeros(1, controller_under_test.dynamics_model.n_dims)
        else:
            default_state = self.default_state

        default_state = default_state.type_as(x_vals)

        # Make a copy of the default state, which we'll modify on every loop
        x = (
            default_state.clone()
            .detach()
            .reshape(1, controller_under_test.dynamics_model.n_dims)
        )

        print(x.shape)

        # Loop through the grid
        prog_bar_range = tqdm.trange(self.n_grid, desc="Plotting CBF", leave=True)
        for i in prog_bar_range:
            for j in range(self.n_grid):
                # Adjust x to be at the current grid point
                x[:, self.x_axis_index] = x_vals[i]
                x[:, self.y_axis_index] = y_vals[j]
                # x[:, 2] = -0.4
                # x[:, 3] = 0.0
                # x[:, 4] = 0.4
                # x[:, 5] = 0.0
                # x[:, 6] = 3.14 
                # x[:, 7] = 0.7854
                # x[:, 8] = 2.3562

                # x[:, 2:] = torch.tensor([0.5293, -0.2825, -0.4753, -0.3662,  1.5841,  2.6514, 0.6565])

                # print(x)
                # quit()
                
                # x = controller_under_test.dynamics_model.states_rel(x)

                # Get the value of the CBF
                h = controller_under_test.V(x)

                # Get the goal, safe, or unsafe classification
                is_goal = controller_under_test.dynamics_model.goal_mask(x).all()
                is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
                is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()

                # Store the results
                results.append(
                    {
                        self.x_axis_label: x_vals[i].cpu().numpy().item(),
                        self.y_axis_label: y_vals[j].cpu().numpy().item(),
                        "CBF Value (h)": h.cpu().numpy().item(),
                        "Goal region": is_goal.cpu().numpy().item(),
                        "Safe region": is_safe.cpu().numpy().item(),
                        "Unsafe region": is_unsafe.cpu().numpy().item(),
                    }
                )

        return pd.DataFrame(results)

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
        print(results_df)
        quit()
        # Set consistent styling and theme
        sns.set_theme(context="talk", style="white")
        plt.rc('font', family='P052', size=18)  # Consistent font family and size
        plt.rc('axes', titlesize=18)  # Title font size
        plt.rc('axes', labelsize=18)  # Axes label font size
        plt.rc('xtick', labelsize=18)  # X-tick font size
        plt.rc('ytick', labelsize=18)  # Y-tick font size

        # Plot a contour of h
        fig, ax = plt.subplots(1, 1)
        # fig.set_size_inches(12, 8)
        ax.set_aspect('equal')

        # Set axis limits and labels
        ax.set_title(f"$\\lambda = 1$")
        xl, xr = -0.75, 0.75
        yl, yr = -0.75, 0.75
        ax.set_xlim(xl, xr)
        ax.set_ylim(yl, yr)
        ax.set_xlabel(f"{self.x_axis_label}", fontweight='bold', labelpad=-15)
        ax.set_ylabel(f"{self.y_axis_label}", fontweight='bold', labelpad=-25)

        xticks = np.linspace(xl, xr, num=2)  # Adjust num to control the number of ticks
        yticks = np.linspace(yl, yr, num=2)  # Adjust num to control the number of ticks
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)


        # Define consistent contour levels
        value_min = results_df["CBF Value (h)"].min()
        value_max = results_df["CBF Value (h)"].max()
        contour_levels = np.linspace(-0.5, 0.5, 21)  # Same intervals as before

        # Plot contours
        contours = ax.tricontourf(
            results_df[self.x_axis_label],
            results_df[self.y_axis_label],
            results_df["CBF Value (h)"] * -1,
            levels=contour_levels,
            cmap='coolwarm',  # Match color map
        )

        # Add color bar
        cbar = fig.colorbar(contours, ax=ax, orientation="vertical")
        # cbar.ax.set_ylabel("CBF Value (h)", rotation=270, labelpad=20)

        # Overlay safe/unsafe regions if specified
        if self.plot_safe_region:
            ax.plot([], [], c="green", label="Safe Region")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Safe region"],
                colors=["green"],
                levels=[0.5],
            )
            ax.plot([], [], c="magenta", label="Unsafe Region")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Unsafe region"],
                colors=["magenta"],
                levels=[0.5],
            )

        # Add legend
        # ax.legend(loc="upper right")

        fig_handle = ("CBF Contour", fig)

        # Save and/or display plots
        if display_plots:
            plt.savefig("contours.png")
            plt.show()
            return []
        else:
            return [fig_handle]