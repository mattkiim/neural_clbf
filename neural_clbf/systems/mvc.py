from .control_affine_system import ControlAffineSystem
import torch
import math
import numpy as np
from typing import Optional, Tuple
from abc import abstractmethod
from neural_clbf.systems.utils import Scenario, lqr, ScenarioList
import matplotlib.pyplot as plt

class MultiVehicleCollision(ControlAffineSystem):
    def __init__(
        self, 
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        # Define vehicle-specific parameters 
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        self.obs_dim = 9

        
        super().__init__(
            nominal_params=nominal_params,
            dt=0.01,
            controller_dt=controller_dt,
            scenarios=scenarios
        )
    
    @property
    def n_dims(self) -> int:
        return 9  # Three vehicles with (x, y, theta) each

    @property
    def n_controls(self) -> int:
        return 3  # Control inputs for each vehicle's angular velocity

    @property
    def angle_dims(self) -> list:
        return [6, 7, 8]  # Indices for theta angles of each vehicle

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_limits = torch.tensor([-1, -1, -1, -1, -1, -1, -math.pi, -math.pi, -math.pi])
        upper_limits = torch.tensor([1, 1, 1, 1, 1, 1, math.pi, math.pi, math.pi])

        return (upper_limits, lower_limits)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_limits = torch.tensor([-self.omega_max] * self.n_controls)
        upper_limits = torch.tensor([self.omega_max] * self.n_controls)
        return (upper_limits, lower_limits)

    def validate_params(self, params: dict) -> bool:
        # Validation checks for system parameters
        return all(param in params for param in ['velocity', 'omega_max', 'collisionR'])

    def _f(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        MattKiim: NON-relative Control-independent dynamics f(x). Not being used.
        """
        # Extract batch size
        batch_size = x.shape[0]
        
        # Initialize f with zeros, shape [batch_size, n_dims, 1]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)  # Ensure the tensor is of the same dtype as the input x

        # Extract the parameters (velocity is the key control parameter here)
        velocity = params['velocity']

        # Dynamics for each vehicle (f1, f2, f3 for x, y components and theta)
        f[:, 0, 0] = velocity * torch.cos(x[:, 6])  # Vehicle 1 x-velocity
        f[:, 1, 0] = velocity * torch.sin(x[:, 6])  # Vehicle 1 y-velocity
        f[:, 2, 0] = velocity * torch.cos(x[:, 7])  # Vehicle 2 x-velocity
        f[:, 3, 0] = velocity * torch.sin(x[:, 7])  # Vehicle 2 y-velocity
        f[:, 4, 0] = velocity * torch.cos(x[:, 8])  # Vehicle 3 x-velocity
        f[:, 5, 0] = velocity * torch.sin(x[:, 8])  # Vehicle 3 y-velocity

        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        MattKiim: the NON-RELATIVE control-dependent dynamics. Not currently being used. 
        """
        g = torch.zeros(x.shape[0], self.n_dims, self.n_controls)
        g[:, 6, 0] = 1.0
        g[:, 7, 1] = 1.0
        g[:, 8, 2] = 1.0
        return g

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Defines safe regions based on collision distance
        return self.boundary_fn(x) > 0 + 0.2

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Unsafe region if boundary function is less than zero
        return self.boundary_fn(x) < 0

    def boundary_fn(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the collision boundary function based on the minimum pairwise distance 
        between three vehicles.

        args:
            state: (batch_size, 9) tensor representing (x1, y1, x2, y2, x3, y3, θ1, θ2, θ3)
        returns:
            boundary_values: (batch_size,) tensor representing the closest safe distance
                            before a collision.
        """
        # Extract vehicle positions
        xy1 = state[:, 0:2]  # (x1, y1)
        xy2 = state[:, 2:4]  # (x2, y2)
        xy3 = state[:, 4:6]  # (x3, y3)

        # Compute pairwise distances
        dist_12 = torch.norm(xy1 - xy2, dim=-1) - self.collisionR
        dist_13 = torch.norm(xy1 - xy3, dim=-1) - self.collisionR
        dist_23 = torch.norm(xy2 - xy3, dim=-1) - self.collisionR

        # Return the **minimum pairwise distance**
        boundary_values = torch.min(torch.min(dist_12, dist_13), dist_23)

        return boundary_values


    

    def u_nominal2(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters. The nominal controller is LQR.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        
        # self.goal_point = self.ref_x 
        # # FIXME : What should the goal point be? Depends on i/c of cars. Currently, aiming to crash at origin. 
        # # Probably should be on far side of circle for each agent.

        # # Compute nominal control from feedback + equilibrium control
        # u_nominal = -(self.K.type_as(x) @ (x - self.goal_point).T).T 
        # u_eq = torch.zeros_like(u_nominal)
        # u = u_nominal + u_eq

        # # Clamp given the control limits
        # upper_u_lim, lower_u_lim = self.control_limits
        # for dim_idx in range(self.n_controls):
        #     u[:, dim_idx] = torch.clamp(
        #         u[:, dim_idx],
        #         min=lower_u_lim[dim_idx].item(),
        #         max=upper_u_lim[dim_idx].item(),
        #     )

        return self.ref_u
    
    def u_nominal(self, x: torch.Tensor, params: Optional[Scenario] = None) -> torch.Tensor:
        """
        A simple nominal controller for three Dubins vehicles stacked in one state.
        Each vehicle has (x, y, theta) with fixed forward speed and an angular rate as control.
        
        We steer each vehicle toward the origin (0, 0).
        """
        # x shape: [batch_size, 9], grouped as:
        #  vehicle1: (x0, y0, theta0) at indices 0,1,6
        #  vehicle2: (x1, y1, theta1) at indices 2,3,7
        #  vehicle3: (x2, y2, theta2) at indices 4,5,8

        batch_size = x.shape[0]
        u = torch.zeros(batch_size, self.n_controls, device=x.device, dtype=x.dtype)
        # u will be [batch_size, 3], each column is the angular velocity for vehicle i.

        # gain for heading control
        k_p = 1.0  

        for i in range(3):
            # Indices for (x_i, y_i, theta_i)
            x_idx = 2 * i      # 0->(x0,y0), 1->(x1,y1), 2->(x2,y2)
            y_idx = 2 * i + 1
            theta_idx = 6 + i  # angles start at 6,7,8

            x_i = x[:, x_idx]       # shape [batch_size]
            y_i = x[:, y_idx]       # shape [batch_size]
            theta_i = x[:, theta_idx]

            desired_heading = torch.atan2(-y_i, -x_i)

            heading_error = desired_heading - theta_i
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

            w_cmd = k_p * heading_error
            w_cmd = torch.clamp(w_cmd, -self.omega_max, self.omega_max)

            # Assign the i-th control
            u[:, i] = w_cmd

        return u