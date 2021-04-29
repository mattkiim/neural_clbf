"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, lqr


class InvertedPendulum(ControlAffineSystem):
    """
    Represents a damped inverted pendulum.

    The system has state

        x = [theta, theta_dot]

    representing the angle and velocity of the pendulum, and it
    has control inputs

        u = [u]

    representing the torque applied.

    The system is parameterized by
        m: mass
        L: length of the pole
        b: damping
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1

    # State indices
    THETA = 0
    THETA_DOT = 1
    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Requires keys ["m", "L", "b"]
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(nominal_params, dt)

        # Compute the LQR gain matrix for the nominal parameters
        # Linearize the system about the x = 0, u = 0
        A = np.zeros((self.n_dims, self.n_dims))
        A[0, 1] = 1.0
        A[1, 0] = grav / self.nominal_params["L"]
        A[1, 1] = -self.nominal_params["b"] / (
            self.nominal_params["m"] * self.nominal_params["L"] ** 2
        )

        B = np.zeros((self.n_dims, self.n_controls))
        B[1, 0] = 1.0 / (self.nominal_params["m"] * self.nominal_params["L"] ** 2)

        # Adapt for discrete time
        if controller_dt is None:
            controller_dt = dt

        A = np.eye(self.n_dims) + controller_dt * A
        B = controller_dt * B

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        self.K = torch.tensor(lqr(A, B, Q, R))

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
                    Requires keys ["m", "L", "b"]
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "m" in params
        valid = valid and "L" in params
        valid = valid and "b" in params

        # Make sure all parameters are physically valid
        valid = valid and params["m"] > 0
        valid = valid and params["L"] > 0
        valid = valid and params["b"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return InvertedPendulum.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return [InvertedPendulum.THETA]

    @property
    def n_controls(self) -> int:
        return InvertedPendulum.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[InvertedPendulum.THETA] = 2.0
        upper_limit[InvertedPendulum.THETA_DOT] = 2.0

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([10.0, 10.0])
        lower_limit = -torch.tensor([10.0, 10.0])

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Avoid angles that are too large
        theta_max_safe = np.pi / 4
        angle_small_enough = x[:, InvertedPendulum.THETA].abs() <= theta_max_safe
        safe_mask.logical_and_(angle_small_enough)

        # Avoid speeds that are too large
        dtheta_max_safe = 1.5
        speed_small_enough = x[:, InvertedPendulum.THETA_DOT].abs() <= dtheta_max_safe
        safe_mask.logical_and_(speed_small_enough)

        # Also don't let angles go too negative
        angle_positive = x[:, InvertedPendulum.THETA] >= -0.1
        safe_mask.logical_and_(angle_positive)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Avoid angles that are too large
        theta_min_unsafe = np.pi / 3
        angle_too_big = x[:, InvertedPendulum.THETA].abs() >= theta_min_unsafe
        unsafe_mask.logical_or_(angle_too_big)

        # Avoid speeds that are too large
        dtheta_min_unsafe = 1.7
        speed_too_big = x[:, InvertedPendulum.THETA_DOT].abs() >= dtheta_min_unsafe
        unsafe_mask.logical_or_(speed_too_big)

        # Also don't let angles go too negative
        angle_negative = x[:, InvertedPendulum.THETA] <= -0.15
        unsafe_mask.logical_or_(angle_negative)

        return unsafe_mask

    def distance_to_goal(self, x: torch.Tensor) -> torch.Tensor:
        """Return the distance from each point in x to the goal (positive for points
        outside the goal, negative for points inside the goal), normalized by the state
        limits.

        args:
            x: the points from which we calculate distance
        """
        upper_limit, _ = self.state_limits
        return x.norm(dim=-1) / upper_limit.norm()

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        goal_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Define the goal region as being near the goal
        near_goal = x.norm(dim=-1) <= 0.2
        goal_mask.logical_and_(near_goal)

        # The goal set has to be a subset of the safe set
        goal_mask.logical_and_(self.safe_mask(x))

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract the needed parameters
        m, L, b = params["m"], params["L"], params["b"]
        # and state variables
        theta = x[:, InvertedPendulum.THETA]
        theta_dot = x[:, InvertedPendulum.THETA_DOT]

        # The derivatives of theta is just its velocity
        f[:, InvertedPendulum.THETA, 0] = theta_dot

        # Acceleration in theta depends on theta via gravity and theta_dot via damping
        f[:, InvertedPendulum.THETA_DOT, 0] = (
            grav / L * torch.sin(theta) - b / (m * L ** 2) * theta_dot
        )

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # Extract the needed parameters
        m, L = params["m"], params["L"]

        # Effect on theta dot
        g[:, InvertedPendulum.THETA_DOT, InvertedPendulum.U] = 1 / (m * L ** 2)

        return g

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters. For the inverted
        pendulum, the nominal controller is LQR

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nominal: bs x self.n_controls tensor of controls
        """
        # Compute nominal control from feedback + equilibrium control
        u_nominal = -(self.K.type_as(x) @ (x - self.goal_point.squeeze()).T).T
        u_eq = torch.zeros_like(u_nominal)

        return u_nominal + u_eq