from .control_affine_system import ControlAffineSystem
import torch
import numpy as np
import math
from typing import Optional, Tuple
from abc import abstractmethod
from neural_clbf.systems.utils import Scenario, lqr

class MultiVehicleCollisionRelative(ControlAffineSystem):
    def __init__(self, dt: float = 0.01):
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
            dt=dt,
        )

        ## Compute LQR Gain Matrix

        self.controller_dt = dt

        A = np.zeros((self.n_dims, self.n_dims))
        B = np.zeros((self.n_dims, self.n_controls))

        # Fill A and B

        ref_x = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Should this be relative to a particular i/c? or crash state?
        ref_u = torch.tensor([0.0, 0.0, 0.0])       # zeros seem correct here
        self.ref_x = ref_x

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ref_u = ref_u.to("cpu")

        v = self.velocity
        
        A[0, 1], A[0, 6] =  ref_u[0], -v * np.sin(ref_x[6]) # dot x1_l
        A[1, 0], A[1, 6] = -ref_u[0],  v * np.cos(ref_x[6]) # dot y1_l
        A[2, 3], A[2, 7] =  ref_u[0], -v * np.sin(ref_x[7]) # dot x2_l
        A[2, 2], A[2, 7] = -ref_u[0],  v * np.cos(ref_x[7]) # dot y2_l
        A[4, :] = A[0, :] - A[2, :] # dot x3_l
        A[5, :] = A[1, :] - A[3, :] # dot y3_l

        B[0, 0] =  ref_x[1]
        B[1, 0] = -ref_x[0]
        B[2, 0] =  ref_x[3]
        B[3, 0] = -ref_x[2]
        B[4, 0] = B[0, 0] - B[2, 0]
        B[5, 0] = B[1, 0] - B[3, 0]
        B[6, 0], B[6, 1] = -1, 1
        B[7, 0], B[7, 2] = -1, 1
        B[8, :] = B[6, :] - B[7, :]

        # Discretize A and B
        A = np.eye(self.n_dims) + self.controller_dt * A
        B = self.controller_dt * B

        print(A, B)

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        # self.K = torch.tensor(lqr(A, B, Q, R))
    
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
        lower_limits = torch.tensor([-2, -2, -2, -2, -2, -2, -math.pi, -math.pi, -math.pi])
        upper_limits = torch.tensor([2, 2, 2, 2, 2, 2, math.pi, math.pi, math.pi])
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
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system.
                    If None, defaults to the nominal parameters used at initialization.
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and initialize the result tensor
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract velocity parameter
        v = params["velocity"]

        # Extract state variables
        x1r, y1r, theta1 = x[:, 0], x[:, 1], x[:, 6]
        x2r, y2r, theta2 = x[:, 2], x[:, 3], x[:, 7]

        # Relative dynamics for x1r, y1r
        f[:, 0, 0] = v * torch.cos(theta1) - v
        f[:, 1, 0] = v * torch.sin(theta1)

        # Relative dynamics for x2r, y2r
        f[:, 2, 0] = v * torch.cos(theta2) - v
        f[:, 3, 0] = v * torch.sin(theta2)

        # Relative dynamics for x3r, y3r (difference between vehicle 1 and vehicle 2 dynamics)
        f[:, 4, 0] = f[:, 0, 0] - f[:, 2, 0]
        f[:, 5, 0] = f[:, 1, 0] - f[:, 3, 0]

        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Return the control-dependent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system.
                    If None, defaults to the nominal parameters used at initialization.
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and initialize the result tensor
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Extract state variables
        x1r, y1r = x[:, 0], x[:, 1]
        x2r, y2r = x[:, 2], x[:, 3]

        # Control terms for x1r, y1r
        g[:, 0, 0] = y1r  # x1r_dot w.r.t u1
        g[:, 1, 0] = -x1r  # y1r_dot w.r.t u1

        # Control terms for x2r, y2r
        g[:, 2, 1] = y2r  # x2r_dot w.r.t u2
        g[:, 3, 1] = -x2r  # y2r_dot w.r.t u2

        # Control terms for x3r, y3r (difference of above)
        g[:, 4, 0] = g[:, 0, 0] - g[:, 2, 1]  # x3r_dot w.r.t u1, u2
        g[:, 4, 1] = -g[:, 2, 1]
        g[:, 5, 0] = g[:, 1, 0] - g[:, 3, 1]  # y3r_dot w.r.t u1, u2
        g[:, 5, 1] = -g[:, 3, 1]

        # Control terms for relative orientation dynamics
        g[:, 6, 0] = -1.0  # theta1_dot w.r.t u1
        g[:, 7, 1] = -1.0  # theta2_dot w.r.t u2
        g[:, 8, 0] = -1.0  # theta3_dot w.r.t u1
        g[:, 8, 1] = 1.0   # theta3_dot w.r.t u2

        return g


    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Defines safe regions based on collision distance
        return self.boundary_fn(x) > 0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Unsafe region if boundary function is less than zero
        return self.boundary_fn(x) < 0
    
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
    

    def states_rel(self, x):
        x_rel = x.clone()
        x_rel[:, 0:2] = x[:, 0:2] - x[:, 2:4]
        x_rel[:, 2:4] = x[:, 4:6] - x[:, 0:2]
        x_rel[:, 4:6] = x_rel[:, 0:2] - x_rel[:, 2:4]
        
        return x_rel