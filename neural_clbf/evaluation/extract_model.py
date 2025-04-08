import torch
import matplotlib
matplotlib.use('TkAgg')

import glob
import os

from neural_clbf.controllers import NeuralCBFController


# checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_56/checkpoints/" # best
# checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_58/checkpoints/" # gamma=0.5
checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_61/checkpoints/" # r=0.4
checkpoint_dir = "/home/ubuntu/neural_clbf_mk/neural_clbf/training/logs/multivehicle_collision/commit_c69834e/version_67/checkpoints/" # r=0.4


ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

# Select the latest checkpoint file and store it in log_file
log_file = max(ckpt_files, key=os.path.getctime) if ckpt_files else None

neural_controller = NeuralCBFController.load_from_checkpoint(log_file)


# Load the full PyTorch Lightning checkpoint
checkpoint = torch.load(log_file, map_location=torch.device("cpu"))

# Extract and save only the model state_dict (weights)
torch.save(checkpoint["state_dict"], "model.pth")

print("Model weights extracted and saved as model.pth")
