import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        return self.linear_2(self.activation(self.linear_1(x)))
