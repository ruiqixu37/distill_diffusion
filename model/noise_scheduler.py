import torch
import torch.nn as nn
import numpy as np


class NoiseSchedule_Fixed(nn.Module):
    def __init__(self, config):
        super(NoiseSchedule_Fixed, self).__init__()

        self.config = config

    def forward(self, t):
        return torch.cos(t * np.pi / 2) ** 2


class NoiseSchedule_Scalar(nn.Module):
    """
    see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L326C34-L326C34
    """
    def __init__(self, config):
        super(NoiseSchedule_Scalar, self).__init__()

        self.gamma_min = config['gamma_min']
        self.gamma_max = config['gamma_max']

        init_bias = self.gamma_min
        init_scale = self.gamma_max - init_bias

        # Define the weights as Parameters so they are trainable
        self.w = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def post_process(self, unbounded_gamma):
        gamma_0 = -np.log(self.gamma_max)
        gamma_1 = -np.log(self.gamma_min)

        bounded_gamma = gamma_0 + (gamma_1 - gamma_0) * \
            (unbounded_gamma - self(0)) / (self(1) - self(0))

        return bounded_gamma

    def forward(self, t):
        bounded_gamma = self.post_process(self.b + torch.abs(self.w) * t)
        alpha_sq = 1 - torch.sigmoid(bounded_gamma)

        return alpha_sq
