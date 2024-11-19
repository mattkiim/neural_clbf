from .control_affine_system import ControlAffineSystem
import torch
import math
from typing import Optional, Tuple
from abc import abstractmethod

class MultiVehicleCollision(ControlAffineSystem):
    def __init__(self):
        # Define vehicle-specific parameters 
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        self.obs_dim = 9
        
        # Define state properties
        nominal_params = {
            'velocity': self.velocity,
            'omega_max': self.omega_max,
            'collisionR': self.collisionR
        }
        super().__init__(
            nominal_params=nominal_params,
            dt=0.01,
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

    def _f_nr(self, x: torch.Tensor, params: dict) -> torch.Tensor:
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
        # Assuming x[:, 6], x[:, 7], x[:, 8] are the theta values for the 3 vehicles
        f[:, 0, 0] = velocity * torch.cos(x[:, 6])  # Vehicle 1 x-velocity
        f[:, 1, 0] = velocity * torch.sin(x[:, 6])  # Vehicle 1 y-velocity
        f[:, 2, 0] = velocity * torch.cos(x[:, 7])  # Vehicle 2 x-velocity
        f[:, 3, 0] = velocity * torch.sin(x[:, 7])  # Vehicle 2 y-velocity
        f[:, 4, 0] = velocity * torch.cos(x[:, 8])  # Vehicle 3 x-velocity
        f[:, 5, 0] = velocity * torch.sin(x[:, 8])  # Vehicle 3 y-velocity

        return f

    def _f(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Relative control-independent dynamics f(x)"""
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), dtype=x.dtype, device=x.device)

        # Velocity
        velocity = params['velocity']

        # Relative dynamics
        # Vehicle 1 w.r.t. 2
        f[:, 0, 0] = velocity * (torch.cos(x[:, 3]) - torch.cos(x[:, 0]))  # x_1,2
        f[:, 1, 0] = velocity * (torch.sin(x[:, 3]) - torch.sin(x[:, 0]))  # y_1,2

        # Vehicle 2 w.r.t. 3
        f[:, 2, 0] = velocity * (torch.cos(x[:, 6]) - torch.cos(x[:, 3]))  # x_2,3
        f[:, 3, 0] = velocity * (torch.sin(x[:, 6]) - torch.sin(x[:, 3]))  # y_2,3

        # Vehicle 1 w.r.t. 3
        f[:, 4, 0] = velocity * (torch.cos(x[:, 6]) - torch.cos(x[:, 0]))  # x_1,3
        f[:, 5, 0] = velocity * (torch.sin(x[:, 6]) - torch.sin(x[:, 0]))  # y_1,3

        return f

    def _g_nr(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        MattKiim: the NON-RELATIVE control-dependent dynamics. Not currently being used. 
        """
        g = torch.zeros(x.shape[0], self.n_dims, self.n_controls)
        g[:, 6, 0] = 1.0
        g[:, 7, 1] = 1.0
        g[:, 8, 2] = 1.0
        return g

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Relative control-dependent dynamics g(x)"""
        g = torch.zeros((x.shape[0], self.n_dims, self.n_controls), dtype=x.dtype, device=x.device)

        # Vehicle 1 w.r.t. 2
        g[:, 6, 0] = 1.0  # u_1 - u_2 for relative orientation
        g[:, 7, 1] = 1.0  # u_2 - u_3 for relative orientation
        g[:, 8, 2] = 1.0  # u_1 - u_3 for relative orientation

        return g

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Defines safe regions based on collision distance
        return self.boundary_fn(x) > 0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Unsafe region if boundary function is less than zero
        return self.boundary_fn(x) < 0

    def boundary_fn_nr(self, state: torch.Tensor) -> torch.Tensor:
        """
        MattKiim: the NON-RELATIVE boundary function (original). Not currently being used. 
        """
        # Computes the minimum distance to maintain safety (collision avoidance)
        boundary_values = torch.norm(state[:, 0:2] - state[:, 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[:, 0:2] - state[:, 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Check collision between other vehicles
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[:, evader1_coords_index:evader1_coords_index+2] - state[:, evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values
    
    def boundary_fn(self, state: torch.Tensor) -> torch.Tensor:
        # Relative distances
        dist_1_2 = torch.norm(state[:, 0:2], dim=-1)  # Relative distance between 1 and 2
        dist_2_3 = torch.norm(state[:, 2:4], dim=-1)  # Relative distance between 2 and 3
        dist_1_3 = torch.norm(state[:, 4:6], dim=-1)  # Relative distance between 1 and 3

        # Safety condition
        boundary_values = torch.min(
            torch.min(dist_1_2, dist_2_3),
            dist_1_3
        ) - self.collisionR

        return boundary_values