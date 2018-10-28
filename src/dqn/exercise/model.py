import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.model = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=state_size, out_features=16, bias=True)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(in_features=16, out_features=8, bias=True)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(in_features=8, out_features=action_size, bias=True))]))


    def forward(self, state):
        """Build a network that maps state -> action values."""
        action_values = self.model.forward(state)
        return action_values
