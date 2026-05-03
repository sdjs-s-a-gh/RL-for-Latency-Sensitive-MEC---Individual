import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        """
            Constructor for the Feed Forward Neural Network (FFNN).
            
            This constructor creates a 3-layer FFNN that uses an
            ReLU activation function.
            
            @param input_dimensions (int): The input dimensions / number of state variables
            the PPO agent will observe at a given time.
            @param output_dimensions (int): The output dimensions of the FFNN, which is either
            the number of actions the PPO agent will make (for the Policy network) or 1 (for
            the value network). 
        """
        super(FeedForwardNeuralNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dimensions, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dimensions)

    def forward(self, state):
        """
            Performs a forward pass on the network and returns
            an output.
            
            @param state (array): An array that includes the following state variables
            in this order:
            - Required CPU Cycles: float
            - Communication Latency: float
            - Resource Utilisation: float
            - Queue Length: float
            - Total Queue Cycles: float
            
            @return: Either a mean action or a value depending on what network calls
            this function.
        """
        # Convert the state to a tensor if it is not one already.
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float)
            
        activation1 = F.relu(self.layer1(state))
        activation2 = F.relu(self.layer2(activation1))

        """
        The output of the neural network at the final layer. In this scenario, this value
        may be the mean action for the policy/actor network or the value for the
        value/critic network. 
        """
        output = self.layer3(activation2)

        return output

