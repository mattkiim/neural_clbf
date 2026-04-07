import numpy as np
import torch

def print_v_and_grads(neural_controller, x, params=None, name=""):
    """
    neural_controller: instance of your NeuralCBFController (loaded from ckpt)
    x: (bs, 9) tensor: [x1,y1,x2,y2,x3,y3,th1,th2,th3]
    params: dict scenario for dynamics_model._f/_g if needed
    """
    neural_controller.eval()
    device = next(neural_controller.parameters()).device
    x = x.to(device).float()

    # V and Jacobian w.r.t. ORIGINAL 9D state
    V, JV = neural_controller.V_with_jacobian(x)   # V: (bs,1), JV: (bs,1,9)

    # Print first element
    i = 0
    x0 = x[i].detach().cpu().numpy()
    V0 = V[i].detach().cpu().numpy().squeeze()
    dVdx0 = JV[i, 0].detach().cpu().numpy()

    np.set_printoptions(precision=6, suppress=True)
    print("\n==============================")
    if name:
        print(f"[{name}]")
    print("x =", x0)
    print("V(x) =", float(V0))
    print("dV/dx =", dVdx0)
    print("dV/dtheta =", dVdx0[6:9])

    # If dynamics provides _f and _g, print Lie derivatives too
    dyn = neural_controller.dynamics_model
    if params is not None and hasattr(dyn, "_f") and hasattr(dyn, "_g"):
        with torch.no_grad():
            f = dyn._f(x, params)  # (bs,9,1)
            g = dyn._g(x, params)  # (bs,9,3)

            LfV = torch.bmm(JV, f).squeeze(-1).squeeze(-1)  # (bs,)
            LgV = torch.bmm(JV, g).squeeze(1)               # (bs,3)

        print("LfV =", float(LfV[i].cpu().item()))
        print("LgV =", LgV[i].detach().cpu().numpy())

        # "bang-bang" omega suggested by your u()
        omega_max = float(params.get("omega_max", 1.1))
        u_star = omega_max * np.sign(dVdx0[6:9])
        print("u_star = omega_max*sign(dV/dtheta) =", u_star)
    else:
        print("(LfV/LgV not printed: pass params and ensure dynamics has _f/_g)")
        

import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import glob
import os


from neural_clbf.controllers import NeuralCLBFController, NeuralCBFController


import matplotlib.pyplot as plt
from neural_clbf.experiments import (
    ExperimentSuite,
    CBFContourExperiment,
    RolloutStateSpaceExperiment,
    RolloutSuccessRateExperiment,
)


start_x = torch.tensor(
    [
        [0.0080, -0.5999,  0.5293, -0.2825, -0.4753, -0.3662, 1.5841, 2.6514, 0.6565],
    ]
)

start_x = torch.tensor(
    [
        [-1.3675, -0.8711, -0.9627, -0.4747, 0.1088, -0.7970, -1.569, -1.081, 2.488],
    ]
)

file_path = 'boundary_initials.npy'
# file_path = 'initial_conditions_2.npy'

initial_conditions = np.load(file_path)

# print(initial_conditions.shape); quit()
# start_x = torch.tensor([initial_conditions[8, :-1]])
# print(start_x); quit()

start_xs = torch.tensor(initial_conditions[:, :-1], dtype=torch.float32)

nominal_params = {"angle_alpha_factor": 1.2, "velocity": 0.6, "omega_max": 1.1, "collisionR": 0.25}
scenarios = [
    nominal_params, # add more for robustness
]


def plot_mvc_rel():
    # checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_0ce2993/version_4/checkpoints/" # gamma=0.0
    checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c00856e/version_1/checkpoints/" # gamma=0.0

    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    # Select the latest checkpoint file and store it in log_file
    log_file = max(ckpt_files, key=os.path.getctime) if ckpt_files else None

    neural_controller = NeuralCBFController.load_from_checkpoint(log_file)
    
    neural_controller = neural_controller.to("cpu")  # or "cuda"
    params = nominal_params  # the dict you already have

    print_v_and_grads(neural_controller, start_x.float(), params=params, name="single start_x")
    
    
plot_mvc_rel()