import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        super(FeedForwardNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dimensions, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dimensions)

    def forward(self, state):
        # Convert the state to a tensor if it is not one already.
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float)
            
        activation1 = F.relu(self.layer1(state))
        activation2 = F.relu(self.layer2(activation1))

        """
        The output of the neural network at the final layer. In this scenario, this value
        may be the mean action for the policy/actor network or the value for the
        value/critic network.
        network 
        """
        output = self.layer3(activation2)

        return output

