import torch
import numpy as np

# Load the initial conditions
file_path = "boundary_initials.npy" # on a circle
initial_conditions = np.load(file_path)

# Convert to a PyTorch tensor
x_post = torch.tensor(initial_conditions[:, :], dtype=torch.float32)
# x_post = torch.tensor(initial_conditions2, dtype=torch.float32)

print(x_post.shape)

# Extract the positions
x1, y1 = x_post[:, 0], x_post[:, 1]
x2, y2 = x_post[:, 2], x_post[:, 3]
x3, y3 = x_post[:, 4], x_post[:, 5]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.scatter(x1, y1, c='g', label='Safe', alpha=0.5)
plt.scatter(x2, y2, c='g', label='Safe', alpha=0.5)
plt.scatter(x3, y3, c='g', label='Safe', alpha=0.5)
plt.title("Vehicle 1 Positions (Safe vs Unsafe)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True)
plt.savefig("positions_test.png", dpi=300)
plt.close()


# plot the positions and save to test.png

# Compute pairwise distances
dist_12 = torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)
dist_13 = torch.sqrt((x1 - x3)**2 + (y1 - y3)**2)
dist_23 = torch.sqrt((x2 - x3)**2 + (y2 - y3)**2)

# Find the minimum distance for each state
min_distances = torch.min(torch.stack([dist_12, dist_13, dist_23], dim=1), dim=1).values

# Determine safe and unsafe states
safe_mask = min_distances > 0.25
unsafe_mask = ~safe_mask

# Count safe and unsafe states
safe_count = safe_mask.sum().item()
unsafe_count = unsafe_mask.sum().item()

print(f"Number of safe states: {safe_count}")
print(f"Number of unsafe states: {unsafe_count}")
