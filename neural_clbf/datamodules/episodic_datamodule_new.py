from typing import List, Callable, Tuple, Dict, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from neural_clbf.systems import ControlAffineSystem


class EpisodicDataModule(pl.LightningDataModule):
    """
    DataModule for sampling from a replay buffer
    """

    def __init__(
        self,
        model: ControlAffineSystem,
        initial_domain: List[Tuple[float, float]],
        trajectories_per_episode: int = 100,
        trajectory_length: int = 5000,
        fixed_samples: int = 100000,
        max_points: int = 10000000,
        val_split: float = 0.1,
        batch_size: int = 64,
        quotas: Optional[Dict[str, float]] = None,
    ):
        """Initialize the DataModule."""
        super().__init__()

        self.model = model
        self.n_dims = model.n_dims  # copied for convenience

        # Save the parameters
        self.trajectories_per_episode = trajectories_per_episode
        self.trajectory_length = trajectory_length
        self.fixed_samples = fixed_samples
        self.max_points = max_points
        self.val_split = val_split
        self.batch_size = batch_size
        self.quotas = quotas if quotas is not None else {}

        # Define the sampling intervals for initial conditions as a hyper-rectangle
        assert len(initial_domain) == self.n_dims
        self.initial_domain = initial_domain

        # Save the min, max, central point, and range tensors
        self.x_max, self.x_min = model.state_limits
        self.x_center = (self.x_max + self.x_min) / 2.0
        self.x_range = self.x_max - self.x_min

        # Normal dataset references
        self.training_data = None
        self.validation_data = None

        # Post-dataset references
        self.use_post_dataset = False
        self.post_training_data = None
        self.post_validation_data = None

    # ------------------------------------------------------------------------
    # Standard dataset creation
    # ------------------------------------------------------------------------
    def prepare_data(self):
        """Create the *normal* dataset."""
        # Sample from simulator
        x_sim = self.sample_trajectories(self.model.nominal_simulator)
        # Augment with random samples
        x_sample = self.sample_fixed()
        x = torch.cat((x_sim, x_sample), dim=0)

        # Randomly split data into training and test sets
        random_indices = torch.randperm(x.shape[0])
        val_pts = int(x.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        x_training = x[training_indices]
        x_validation = x[validation_indices]

        # Build datasets
        self.training_data = TensorDataset(
            x_training,
            self.model.goal_mask(x_training),
            self.model.safe_mask(x_training),
            self.model.unsafe_mask(x_training),
        )
        self.validation_data = TensorDataset(
            x_validation,
            self.model.goal_mask(x_validation),
            self.model.safe_mask(x_validation),
            self.model.unsafe_mask(x_validation),
        )

    def sample_trajectories(
        self, simulator: Callable[[torch.Tensor, int], torch.Tensor]
    ) -> torch.Tensor:
        """Generate new data points by simulating trajectories."""
        x_init = torch.Tensor(self.trajectories_per_episode, self.n_dims).uniform_(0.0, 1.0)

        for i in range(self.n_dims):
            min_val, max_val = self.initial_domain[i]
            x_init[:, i] = x_init[:, i] * (max_val - min_val) + min_val

        x_sim = simulator(x_init, self.trajectory_length)
        x_sim = x_sim.view(-1, self.n_dims)
        return x_sim

    def sample_fixed(self) -> torch.Tensor:
        """Generate new data points by sampling (possibly region-constrained)."""
        samples = []
        allocated_samples = 0
        for region_name, quota in self.quotas.items():
            num_samples = int(self.fixed_samples * quota)
            allocated_samples += num_samples

            if region_name == "goal":
                samples.append(self.model.sample_goal(num_samples))
            elif region_name == "safe":
                samples.append(self.model.sample_safe(num_samples))
            elif region_name == "unsafe":
                samples.append(self.model.sample_unsafe(num_samples))
            elif region_name == "boundary":
                samples.append(self.model.sample_boundary(num_samples))

        free_samples = self.fixed_samples - allocated_samples
        assert free_samples >= 0, "Quotas exceed total fixed_samples."
        samples.append(self.model.sample_state_space(free_samples))
        return torch.vstack(samples)

    # ------------------------------------------------------------------------
    # Post-dataset creation
    # ------------------------------------------------------------------------
    def prepare_post_data(self, file_path="s_5000_u_30000_b_1000.npy"):
        """Create the *post* dataset (both train & validation)."""
        # Load the 'post' data
        initial_conditions = np.load(file_path)
        # Slice out the columns you want
        x_post = torch.tensor(initial_conditions[:, :], dtype=torch.float32)

        # (optional) Subsample if too large
        # x_post = x_post[:5000]

        # Random split into train & val
        random_indices = torch.randperm(x_post.shape[0])
        val_pts = int(x_post.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        x_post_training = x_post[training_indices]
        x_post_validation = x_post[validation_indices]

        # Build post-training and post-validation datasets
        self.post_training_data = TensorDataset(
            x_post_training,
            self.model.goal_mask(x_post_training),
            self.model.safe_mask(x_post_training),
            self.model.unsafe_mask(x_post_training),
        )
        self.post_validation_data = TensorDataset(
            x_post_validation,
            self.model.goal_mask(x_post_validation),
            self.model.safe_mask(x_post_validation),
            self.model.unsafe_mask(x_post_validation),
        )

    def switch_to_post_data(self):
        """Flip the flag so that the next calls to train/val_dataloader use the post dataset."""
        self.use_post_dataset = True
        # Make sure we’ve prepared the post data
        if self.post_training_data is None or self.post_validation_data is None:
            self.prepare_post_data()

    # ------------------------------------------------------------------------
    # Common data augmentation (if you ever add more data, etc.)
    # ------------------------------------------------------------------------
    def add_data(self, simulator: Callable[[torch.Tensor, int], torch.Tensor]):
        """
        Augment the training and validation datasets by simulating and sampling
        (for the *normal* dataset).
        """
        # Sample new data
        x_sim = self.sample_trajectories(simulator)
        x_sample = self.sample_fixed()
        x = torch.cat((x_sim, x_sample), dim=0)

        # Randomly split data into training and test sets
        random_indices = torch.randperm(x.shape[0])
        val_pts = int(x.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        # Augment existing data
        new_train = x[training_indices]
        new_val = x[validation_indices]

        # If you have old data in self.x_training, etc., you can do:
        # self.x_training = torch.cat((self.x_training, new_train))
        # ...
        # Then rebuild self.training_data, self.validation_data, etc.
        # Omitted here for brevity

    # ------------------------------------------------------------------------
    # PL.LightningDataModule Hooks
    # ------------------------------------------------------------------------
    def setup(self, stage=None):
        """Called after prepare_data, but before train/val_dataloader. (No-op here)"""
        pass

    def train_dataloader(self) -> DataLoader:
        """Return the correct DataLoader depending on the flag."""
        if self.use_post_dataset:
            return DataLoader(self.post_training_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        """Return the correct DataLoader depending on the flag."""
        if self.use_post_dataset:
            return DataLoader(self.post_validation_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
        else:
            return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
