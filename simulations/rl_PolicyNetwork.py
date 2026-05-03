from rl_FeedForwardNN import FeedForwardNeuralNetwork

import torch
from torch import nn
from torch.distributions import MultivariateNormal 

class PolicyNetwork(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super(PolicyNetwork, self).__init__()

        self.network = FeedForwardNeuralNetwork(input_dimensions, output_dimensions)

        self.log_std = nn.Parameter(torch.zeros(output_dimensions))

    def get_distribution(self, state):
        # Perform a forward pass on the Policy Network to get a mean action.
        mean = self.network(state)

        std = self.log_std.exp()

        covariance_matrix = torch.diag(std **2)

        return MultivariateNormal(mean, covariance_matrix)