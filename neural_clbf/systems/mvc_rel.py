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
        r"""
        Return the "drift" part of the dynamics:  \dot{x} = f(x) + g(x) u.

        In a reference-based approach with constant speed = v:
        - Vehicle 1 (Ego):
            \dot{x1}   = v cos(theta1)
            \dot{y1}   = v sin(theta1)
            \dot{theta1} = 0           (since heading rate is in g(x)*u1)
        - Vehicle 2 relative coords:
            rx2 = x2 - x1
            ry2 = y2 - y1
            => \dot{rx2} = v cos(theta1 + rtheta2) - v cos(theta1)
                \dot{ry2} = v sin(theta1 + rtheta2) - v sin(theta1)
            => \dot{rtheta2} = 0       (since (theta2 - theta1) depends on (u2 - u1))
        - Vehicle 3 is analogous
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), dtype=x.dtype, device=x.device)

        # Unpack states
        x1    = x[:, 0]
        y1    = x[:, 1]
        rx2   = x[:, 2]
        ry2   = x[:, 3]
        rx3   = x[:, 4]
        ry3   = x[:, 5]
        theta1= x[:, 6]
        rtheta2 = x[:, 7]
        rtheta3 = x[:, 8]

        v = params['velocity']

        # 1) Vehicle 1 (Ego)
        # dot{x1} = v cos(theta1)
        # dot{y1} = v sin(theta1)
        f[:, 0, 0] = v * torch.cos(theta1)
        f[:, 1, 0] = v * torch.sin(theta1)
        # dot{theta1} = 0 because heading rate enters via g(x)*u1

        # 2) Vehicle 2 relative coords
        # dot{rx2} = v[cos(theta1 + rtheta2) - cos(theta1)]
        # dot{ry2} = v[sin(theta1 + rtheta2) - sin(theta1)]
        f[:, 2, 0] = v*(torch.cos(theta1 + rtheta2) - torch.cos(theta1))
        f[:, 3, 0] = v*(torch.sin(theta1 + rtheta2) - torch.sin(theta1))
        # dot{rtheta2} = 0, heading difference is in g

        # 3) Vehicle 3 relative coords
        f[:, 4, 0] = v*(torch.cos(theta1 + rtheta3) - torch.cos(theta1))
        f[:, 5, 0] = v*(torch.sin(theta1 + rtheta3) - torch.sin(theta1))
        # dot{rtheta3} = 0

        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        r"""
        The control-affine part:  \dot{x} = f(x) + g(x) u,  with u = [u1, u2, u3].

        We consider:
        - \dot{\theta1}   = u1
        - \dot{rtheta2}   = (u2 - u1)
        - \dot{rtheta3}   = (u3 - u1)
        So g(x) * [u1, u2, u3] should fill these angle derivatives accordingly.

        x1, y1, rx2, ry2, rx3, ry3 have no *direct* linear control terms
        in a standard Dubins model—because the speed v is constant, and
        the heading angles appear in f(x).

        => For the angles we get:

            dot{theta1}   = 1 * u1 + 0 * u2 + 0 * u3
            dot{rtheta2}  = -1 * u1 + 1 * u2 + 0 * u3
            dot{rtheta3}  = -1 * u1 + 0 * u2 + 1 * u3
        """
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls), dtype=x.dtype, device=x.device)

        # Indices: [0,1,2,3,4,5,6,7,8] => x1,y1,rx2,ry2,rx3,ry3,theta1,rtheta2,rtheta3
        # For x1,y1,rx2,ry2,rx3,ry3 => no direct linear control terms
        # For [theta1, rtheta2, rtheta3], define how each is controlled by [u1, u2, u3]

        # dot{theta1}   = u1 => g[:, 6, 0] = 1
        g[:, 6, 0] = 1.0

        # dot{rtheta2}  = u2 - u1 => g[:, 7, 0] = -1,  g[:, 7, 1] = +1
        g[:, 7, 0] = -1.0
        g[:, 7, 1] =  1.0

        # dot{rtheta3}  = u3 - u1 => g[:, 8, 0] = -1, g[:, 8, 2] = +1
        g[:, 8, 0] = -1.0
        g[:, 8, 2] =  1.0

        return g


    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Defines safe regions based on collision distance
        return self.boundary_fn(x) > 0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Unsafe region if boundary function is less than zero
        return self.boundary_fn(x) < 0
    
    def boundary_fn(self, state: torch.Tensor) -> torch.Tensor:
            r"""
            We store:
            (x1, y1) for Ego's absolute position,
            (rx2, ry2) = (x2 - x1, y2 - y1),
            (rx3, ry3) = (x3 - x1, y3 - y1).

            Distances to check:
            - Ego <-> Vehicle2:  ||(rx2, ry2)||
            - Ego <-> Vehicle3:  ||(rx3, ry3)||
            - Vehicle2 <-> Vehicle3:  ||((rx3,ry3) - (rx2,ry2))||
            Then subtract collisionR.
            """
            rx2 = state[:, 2]
            ry2 = state[:, 3]
            rx3 = state[:, 4]
            ry3 = state[:, 5]

            dist_12 = torch.sqrt(rx2**2 + ry2**2)          # Ego->V2
            dist_13 = torch.sqrt(rx3**2 + ry3**2)          # Ego->V3
            dist_23 = torch.sqrt((rx3 - rx2)**2 + (ry3 - ry2)**2)

            # Minimum distance minus collision radius
            boundary_values = torch.min(torch.min(dist_12, dist_13), dist_23) - self.collisionR
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
    

    def states_rel(self, X_abs: torch.Tensor) -> torch.Tensor:
            r"""
            Convert absolute states -> reference-based states.

            Suppose X_abs is shaped (B, 9), storing:
            [x1, y1, x2, y2, x3, y3, theta1, theta2, theta3]
            We want:
            X_rel = [x1, y1, rx2, ry2, rx3, ry3, theta1, rtheta2, rtheta3]

            where:
            rx2 = x2 - x1
            ry2 = y2 - y1
            rx3 = x3 - x1
            ry3 = y3 - y1
            rtheta2 = theta2 - theta1
            rtheta3 = theta3 - theta1
            """
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

            # Indices in X_rel:
            #  [0: x1, 1: y1, 2: rx2, 3: ry2, 4: rx3, 5: ry3, 6: theta1, 7: rtheta2, 8: rtheta3]
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