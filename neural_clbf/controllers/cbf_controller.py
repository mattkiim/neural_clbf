from typing import Tuple

import torch

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.experiments import ExperimentSuite


class CBFController(CLFController):
    """
    A generic CBF-based controller, using the quadratic Lyapunov function found for
    the linearized system to construct a simple barrier function.

    For our purposes, a barrier function h(x): X -> R segments h(safe) <= 0 and
    h(unsafe) >= 0, and dh/dt <= -lambda h(x).

    This definition allows us to re-use a CLF controller. Internally, we'll rename h = V
    with some notational abuse, but let V be negative sometimes.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        experiment_suite: ExperimentSuite,
        cbf_lambda: float = 1.0,
        cbf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            cbf_lambda: scaling factor for the CBF
            cbf_relaxation_penalty: the penalty for relaxing CLF conditions.
            controller_period: the timestep to use in simulating forward Vdot
        """
        super(CBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            clf_lambda=cbf_lambda,
            clf_relaxation_penalty=cbf_relaxation_penalty,
            controller_period=controller_period,
        )

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CBF value and its Jacobian. Remember that we're abusing notation
        and calling our barrier function V

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLF
        returns:
            V: bs tensor of CBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # To make a simple barrier function, use the Lyapunov function shifted down
        V, JV = super().V_with_jacobian(x)
        # V -= 1.0

        return V, JV

    # def solve_CLF_QP(
    #     self,
    #     x: torch.Tensor,
    #     relaxation_penalty: float = None,
    #     u_ref: torch.Tensor = None,
    #     requires_grad: bool = False,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     OVERRIDE the parent's QP-based solver with a bang-bang approach.
    #     We'll interpret the parent's 'V' as 'h(x)' (the barrier).
    #     """
    #     # 1) Compute the barrier function h(x) and its derivatives
    #     Lf_h, Lg_h = self.V_lie_derivatives(x)  

    #     # 2) Get reference control, or fallback to parent's nominal if none provided
    #     if u_ref is None:
    #         u_ref = self.u_reference(x)  # shape: (bs, n_controls)

    #     # 3) Ignore QP and do bang-bang:
    #     bs = x.shape[0]
    #     n_scenarios = self.n_scenarios
    #     n_controls = self.dynamics_model.n_controls

    #     # Example: define a single control limit or use system's control limits
    #     upper_lim, lower_lim = self.dynamics_model.control_limits
    #     # We'll store the final controls in a tensor
    #     u_result = torch.zeros(bs, n_controls, device=x.device, dtype=x.dtype)
    #     # Barrier relaxation is meaningless here, so return zeros
    #     relaxation_result = torch.zeros(bs, 1, device=x.device, dtype=x.dtype)

    #     for i in range(bs):
    #         # Summation across scenarios (heuristic example):
    #         # sum over scenario dimension => shape: (n_controls,)
    #         sum_scenarios = Lg_h[i].sum(dim=0).squeeze(-1)  
    #         # sign(...) => +1 or -1 (or 0)
    #         sign_dir = torch.sign(sum_scenarios)  

    #         # We do a bang-bang by picking either upper_lim or lower_lim
    #         # If sign_dir[j] > 0 => pick upper_lim[j], else lower_lim[j]
    #         # (This is a naive approach; adapt as you see fit.)
    #         u_candidate = torch.where(
    #             sign_dir > 0,
    #             upper_lim.type_as(x),  # pick upper limit
    #             lower_lim.type_as(x),  # pick lower limit
    #         )

    #         u_result[i, :] = u_candidate

    #     return u_result, relaxation_result