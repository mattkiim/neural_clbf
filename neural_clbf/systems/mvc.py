from .control_affine_system import ControlAffineSystem
import torch
import math
from typing import Optional, Tuple
from abc import abstractmethod
from neural_clbf.systems.utils import Scenario, lqr

class MultiVehicleCollision(ControlAffineSystem):
    def __init__(self):
        # Define vehicle-specific parameters 
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        self.obs_dim = 9

        self.ref_u = torch.tensor([0.0, 0.0, 0.0])  
        
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
        # Assuming x[:, 6], x[:, 7], x[:, 8] are the theta values for the 3 vehicles
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
        return self.boundary_fn(x) > 0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Unsafe region if boundary function is less than zero
        return self.boundary_fn(x) < 0

    def boundary_fn(self, state: torch.Tensor) -> torch.Tensor:
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
    

    def u_nominal(
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