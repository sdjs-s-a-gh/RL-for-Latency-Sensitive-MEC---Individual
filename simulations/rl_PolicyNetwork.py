from rl_FeedForwardNN import FeedForwardNeuralNetwork

import torch
from torch import nn
from torch.distributions import MultivariateNormal 

class PolicyNetwork(nn.Module):
    def __init__(self, input_dimensions, output_dimensions):
        """
            Constructs a Feed Forward Neural Network and creates a
            a log standard deviation parameter to allow entropy to
            be modelled.
            
            @param input_dimensions (int): The input dimensions / number of
            state variables the PPO agent will observe at a given time.
            @param output_dimensions (int): The output dimensions of the FFNN / 
            the number of actions the PPO agent will make.
        """
        super(PolicyNetwork, self).__init__()

        self.network = FeedForwardNeuralNetwork(input_dimensions, output_dimensions)

        self.log_std = nn.Parameter(torch.zeros(output_dimensions))

    def get_distribution(self, state):
        """
            Returns a Normal Distribution of the Policy (Actor) network at the
            current time.
        """
        # Perform a forward pass on the Policy Network to get a mean action.
        mean = self.network(state)
        
        # Exponentiate out the log from the log_std
        std = self.log_std.exp()
        
        # Raise the std to the power of 2 to model variance.
        covariance_matrix = torch.diag(std **2)

        return MultivariateNormal(mean, covariance_matrix)