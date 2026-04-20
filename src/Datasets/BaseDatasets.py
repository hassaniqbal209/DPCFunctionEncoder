import torch
from neuromancer import DictDataset
from torch.utils.data import IterableDataset, default_collate


# Used to train the function encoder to model your system
class BaseFunctionEncoderDataset(IterableDataset):
    def __init__(
        self,
        n_points: int,
        n_example_points: int,
        state_size: int,
        action_size: int,
        reference_size: int,
        state_bounds: torch.tensor,
        action_bounds: torch.tensor,
        dt:float,
    ):
        assert state_bounds.shape == (2, state_size), f"State bounds must be a 2D array with shape (2, state_size), got {state_bounds.shape}"
        assert action_bounds.shape == (2, action_size), f"Action bounds must be a 2D array with shape (2, action_size), got {action_bounds.shape}"
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.state_size = state_size
        self.action_size = action_size
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.reference_size = reference_size
        self.dt = dt

    def __iter__(self):
        raise Exception("TODO: Implement this for your class")

    def plot(self, model, args):
        raise Exception("TODO: Implement this for your class")

# Used to train a policy for your system
class BaseTrajectoryDataset(DictDataset):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        state_bounds: torch.tensor,
        action_bounds: torch.tensor,
        initial_state_bounds: torch.tensor,
        dt:float,
    ):
        assert state_bounds.shape == (2, state_size), "State bounds must be a 2D array with shape (2, state_size)"
        assert action_bounds.shape == (2, action_size), "Action bounds must be a 2D array with shape (2, action_size)"
        self.state_size = state_size
        self.action_size = action_size
        self.state_bounds = state_bounds
        self.initial_state_bounds = initial_state_bounds
        self.action_bounds = action_bounds
        self.dt = dt

    def get_policy_training_data(self, num_envs, horizon):
        raise Exception("TODO: Implement this for your class")

    def get_constraints_objectives(self, device):
        raise Exception("TODO: Implement this for your class")

    def plot_trajectory(self, coefficients, cl_system):
        raise Exception("TODO: Implement this for your class")

    def rollout_real_trajectory(self, hidden_parameter, coefficients, policy, save_dir=None):
        raise Exception("TODO: Implement this for your class")

    def collate_fn(self, batch):
        """Wraps the default PyTorch batch collation function and adds a name field.

        :param batch: (dict str: torch.Tensor) dataset sample.
        """
        batch = default_collate(batch)
        batch['name'] = self.name
        return batch