import math
import torch
from typing import Optional, Tuple
from abc import abstractmethod

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList

class Air3D(ControlAffineSystem):
    """
    A single-vehicle Air3D system using the same syntax as MultiVehicleCollision.

    Dynamics (disturbance ignored):
        x_dot    = -v + v cos(psi) + u * y
        y_dot    =  v sin(psi)     - u * x
        psi_dot  =               - u
    """

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        # Extract the parameters from the nominal_params dictionary
        self.collisionR = nominal_params["collisionR"]
        self.velocity = nominal_params["velocity"]
        self.omega_max = nominal_params["omega_max"]
        self.angle_alpha_factor = nominal_params["angle_alpha_factor"]

        super().__init__(
            nominal_params=nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
        )

    @property
    def n_dims(self) -> int:
        """Number of state dimensions: (x, y, psi)."""
        return 3

    @property
    def n_controls(self) -> int:
        """Number of control inputs: a single scalar u for rotation."""
        return 1

    @property
    def angle_dims(self) -> list:
        """Which dimensions are angles (wrapped)? psi is at index 2."""
        return [2]

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Provide upper and lower bounds for (x, y, psi).
        Here we replicate the original [-1, 1] for x/y, and [-π, π] for psi.
        """
        lower_limits = torch.tensor([-1.0, -1.0, -math.pi])
        upper_limits = torch.tensor([1.0,  1.0,  math.pi])
        return (upper_limits, lower_limits)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Control u is constrained to [-omega_max, omega_max].
        """
        lower_limits = torch.tensor([-self.omega_max])
        upper_limits = torch.tensor([ self.omega_max])
        return (upper_limits, lower_limits)

    def validate_params(self, params: dict) -> bool:
        """Checks that required parameters are present."""
        needed = ["collisionR", "velocity", "omega_max", "angle_alpha_factor"]
        return all(k in params for k in needed)

    def _f(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Control-independent dynamics, f(x).
        For each state in the batch, compute:
            f_x   = -v + v cos(psi)
            f_y   =  v sin(psi)
            f_psi =  0
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), device=x.device, dtype=x.dtype)

        v = params["velocity"]
        psi = x[:, 2]

        # f_x
        f[:, 0, 0] = -v + v * torch.cos(psi)
        # f_y
        f[:, 1, 0] = v * torch.sin(psi)
        # f_psi
        f[:, 2, 0] = 0.0

        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Control-dependent dynamics, g(x). We have:
            x_dot   += u * y
            y_dot   += -u * x
            psi_dot += -u
        So each row of g is the partial derivative of [x_dot, y_dot, psi_dot] wrt u.
        """
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls), device=x.device, dtype=x.dtype)

        x_ = x[:, 0]
        y_ = x[:, 1]

        # The single control is u, so shape = [batch_size, 3, 1]
        # g_x(u)   = y
        g[:, 0, 0] = y_
        # g_y(u)   = -x
        g[:, 1, 0] = -x_
        # g_psi(u) = -1
        g[:, 2, 0] = -1.0

        return g

    def boundary_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distance from origin minus collision radius.
        """
        # x, y = x[:, 0], x[:, 1]
        dist_xy = torch.norm(x[:, :2], dim=-1)
        return dist_xy - self.collisionR

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Safe if boundary_fn(x) > 0.
        """
        return self.boundary_fn(x) > 0

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unsafe if boundary_fn(x) < 0.
        """
        return self.boundary_fn(x) < 0

    def u_nominal(self, x: torch.Tensor, params: Optional[Scenario] = None) -> torch.Tensor:
        """
        A simple nominal controller that does nothing (u = 0).
        You could replace this with something more intelligent,
        e.g. steering to reduce distance from the origin.
        """
        batch_size = x.shape[0]
        u = torch.zeros((batch_size, self.n_controls), device=x.device, dtype=x.dtype)

        # Example: Try to steer to reduce x,y error
        #   - Not shown here, but you'd compute a control that helps move
        #     the vehicle to (0, 0) or some other reference.
        # For now, return zero.
        return u
