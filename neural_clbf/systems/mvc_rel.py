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

        # Build nominal params for the parent class
        nominal_params = {
            'velocity': self.velocity,
            'omega_max': self.omega_max,
            'collisionR': self.collisionR
        }
        super().__init__(
            nominal_params=nominal_params,
            dt=dt,
        )

        # A reference / nominal state for linearization (9D)
        # e.g., zero everything or set them to some desired operating point
        self.ref_x = np.zeros(9)
        device = "cuda"
        self.ref_u = torch.tensor(np.zeros(3), dtype=torch.float32).to(device)
        # for instance: [x1=0, y1=0, rx2=0, ry2=0, rx3=0, ry3=0, theta1=0, rtheta2=0, rtheta3=0]

        # Build the continuous-time A, B at the nominal point
        A_ct, B_ct = self.linearize_ego_relative_dynamics(self.ref_x, self.velocity)

        # Discretize using a simple Euler approach
        self.controller_dt = dt
        A_d = np.eye(9) + dt * A_ct
        B_d = dt * B_ct

        # Example cost matrices
        Q = np.eye(9)
        R = np.eye(3)

        # Compute LQR gain (shape: 3x9)
        # If you want K so that  u = -K x  , note that lqr() might return that shape.
        # K_np = lqr(A_d, B_d, Q, R)
        # self.K = torch.tensor(K_np, dtype=torch.float)

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
        lower_limits = torch.tensor([-1, -1, -2, -2, -2, -2, -math.pi, -math.pi, -math.pi])
        upper_limits = torch.tensor([1, 1, 2, 2, 2, 2, math.pi, math.pi, math.pi])
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
        # ... (same as your original drift definition)
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), dtype=x.dtype, device=x.device)

        # Unpack
        x1     = x[:, 0]
        y1     = x[:, 1]
        rx2    = x[:, 2]
        ry2    = x[:, 3]
        rx3    = x[:, 4]
        ry3    = x[:, 5]
        theta1 = x[:, 6]
        rtheta2= x[:, 7]
        rtheta3= x[:, 8]

        v = params["velocity"]

        # Ego
        f[:, 0, 0] = v * torch.cos(theta1)
        f[:, 1, 0] = v * torch.sin(theta1)

        # Vehicle 2 relative
        f[:, 2, 0] = v*(torch.cos(theta1 + rtheta2) - torch.cos(theta1))
        f[:, 3, 0] = v*(torch.sin(theta1 + rtheta2) - torch.sin(theta1))

        # Vehicle 3 relative
        f[:, 4, 0] = v*(torch.cos(theta1 + rtheta3) - torch.cos(theta1))
        f[:, 5, 0] = v*(torch.sin(theta1 + rtheta3) - torch.sin(theta1))

        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        # ... (same as your original control-affine definition)
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls), dtype=x.dtype, device=x.device)

        # Indices: x1,y1,rx2,ry2,rx3,ry3,theta1,rtheta2,rtheta3
        g[:, 6, 0] = 1.0       # dot{theta1} = u1
        g[:, 7, 0] = -1.0      # dot{rtheta2} = u2 - u1 => partial wrt u1 is -1
        g[:, 7, 1] =  1.0
        g[:, 8, 0] = -1.0      # dot{rtheta3} = u3 - u1 => partial wrt u1 is -1
        g[:, 8, 2] =  1.0

        return g

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_fn(x) > 0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary_fn(x) < 0

    def boundary_fn(self, state: torch.Tensor) -> torch.Tensor:
        # ... same logic as before
        rx2 = state[:, 2]
        ry2 = state[:, 3]
        rx3 = state[:, 4]
        ry3 = state[:, 5]

        dist_12 = torch.sqrt(rx2**2 + ry2**2)            
        dist_13 = torch.sqrt(rx3**2 + ry3**2)            
        dist_23 = torch.sqrt((rx3 - rx2)**2 + (ry3 - ry2)**2)

        boundary_values = torch.min(torch.min(dist_12, dist_13), dist_23) - self.collisionR
        return boundary_values

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        # Example: use a linear feedback controller
        # x.shape: (batch, 9)
        # K.shape: (3, 9)
        # x_np = x.detach().cpu().numpy()  # if needed
        # # For demonstration, do a simple linear feedback around the reference point = self.ref_x
        # # u = -K ( x - x_ref ), ignoring any offsets
        # # but your K might be stored as (9 x 3) or (3 x 9); check your shape
        # K_torch = self.K  # (3 x 9) if using lqr from neural_clbf
        # x_err = (x - torch.tensor(self.ref_x, device=x.device, dtype=x.dtype))
        # u_lin = -(K_torch @ x_err.T).T  # shape: (batch, 3)

        # # clamp controls
        # upper_u_lim, lower_u_lim = self.control_limits
        # u_clamped = torch.max(torch.min(u_lin, upper_u_lim), lower_u_lim)

        # return u_clamped
        return self.ref_u

    def states_rel(self, X_abs: torch.Tensor) -> torch.Tensor:
        X_rel = X_abs.clone()
        x1 = X_abs[:, 0]
        y1 = X_abs[:, 1]
        x2 = X_abs[:, 2]
        y2 = X_abs[:, 3]
        x3 = X_abs[:, 4]
        y3 = X_abs[:, 5]
        th1= X_abs[:, 6]
        th2= X_abs[:, 7]
        th3= X_abs[:, 8]

        X_rel[:, 0] = x1
        X_rel[:, 1] = y1
        X_rel[:, 2] = x2 - x1
        X_rel[:, 3] = y2 - y1
        X_rel[:, 4] = x3 - x1
        X_rel[:, 5] = y3 - y1
        X_rel[:, 6] = th1
        X_rel[:, 7] = th2 - th1
        X_rel[:, 8] = th3 - th1

        return X_rel

    def linearize_ego_relative_dynamics(self, x_star: np.ndarray, v: float):
        r"""
        Return the continuous-time A, B for the system around nominal state x_star.

        x_star: shape (9,), storing [x1, y1, rx2, ry2, rx3, ry3, theta1, rtheta2, rtheta3].
        v:      forward speed
        """
        # Unpack angles
        th1 = x_star[6]
        rth2 = x_star[7]  # = theta2 - theta1
        rth3 = x_star[8]  # = theta3 - theta1

        sin_th1    = np.sin(th1)
        cos_th1    = np.cos(th1)
        sin_th1p2  = np.sin(th1 + rth2)
        cos_th1p2  = np.cos(th1 + rth2)
        sin_th1p3  = np.sin(th1 + rth3)
        cos_th1p3  = np.cos(th1 + rth3)

        A = np.zeros((9, 9))

        # row 0: dot{x1} = v cos(th1)
        A[0, 6] = -v * sin_th1

        # row 1: dot{y1} = v sin(th1)
        A[1, 6] =  v * cos_th1

        # row 2: dot{rx2} = v[ cos(th1 + rth2 ) - cos(th1) ]
        A[2, 6] = v * (-sin_th1p2 + sin_th1)
        A[2, 7] = -v * sin_th1p2

        # row 3: dot{ry2} = v[ sin(th1 + rth2 ) - sin(th1) ]
        A[3, 6] = v * (cos_th1p2 - cos_th1)
        A[3, 7] = v * cos_th1p2

        # row 4: dot{rx3} = v[ cos(th1 + rth3 ) - cos(th1) ]
        A[4, 6] = v * (-sin_th1p3 + sin_th1)
        A[4, 8] = -v * sin_th1p3

        # row 5: dot{ry3} = v[ sin(th1 + rth3 ) - sin(th1) ]
        A[5, 6] = v * (cos_th1p3 - cos_th1)
        A[5, 8] = v * cos_th1p3

        # rows 6, 7, 8 are zero (f_6, f_7, f_8 == 0)
        # A[6,:], A[7,:], A[8,:] remain zero

        # B = g(x^*).  In your code, g doesn't depend on x^*, so it's:
        B = np.zeros((9, 3))
        B[6, 0] = 1.0
        B[7, 0] = -1.0
        B[7, 1] =  1.0
        B[8, 0] = -1.0
        B[8, 2] =  1.0

        return A, B
