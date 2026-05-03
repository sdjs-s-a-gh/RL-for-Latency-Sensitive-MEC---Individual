import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class PolicyNetwork(nn.Module):
    """
        A Class representing a 64-in-64-out Feed Forward Neural Network that will act as the 'actor'
        of the PPO resource allocator. This network will decided which action should be taken in 
        the environment, which in this case would be the amount of CPU (as a percentage) to allocate.
    """
    def __init__(self, state_space_dimensions, action_space_dimensions) -> None:
        """
            Parameters:
                state_space_dimensions: the dimensions of the state space from the environment, which
                    is to be used as the input dimension.
                    At the minute, this should be 5. However, as it may change, I have generalised
                    this function to avoid hard-coding any value.
                action_space_dimensions: the dimensions of the state space from the environment, which
                    is to be used as the output dimension.
        """
        super(PolicyNetwork, self).__init__()

        self.layer1 = nn.Linear(state_space_dimensions, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_space_dimensions)       

        #self.log_std = nn.Parameter(torch.zeros(action_space_dimensions)) # is the equivalent of a covariance matrix, but more time efficient as it doesn't produce a matrix (On^2 vs On).
        #self.log_std = nn.Parameter(torch.ones(action_space_dimensions) * -0.5)
        log_std = -0.5 * np.ones(action_space_dimensions, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
    def forward(self, state):
        """
            Completes a forward pass on the Actor/Policy neural network.
            
            Parameters:
                state: The state to use as an input.
            
            Returns:
                mean: The predicted CPU allocation for the forthcoming task.
                log_std: The spread of the Gaussian distribution.
        """
        # Tanh can be substituted for ReLu or Sigmoid.
        activation1 = torch.tanh(self.layer1(state))
        activation2 = torch.tanh(self.layer2(activation1))     
        mean = self.layer3(activation2)

        std = torch.exp(self.log_std) # Exponentiate out the log_std, ensuring that the std is > 0 to not cause any errors.
        #cov_mar = torch.full(size=(1,), fill_value=0.5)
        #std = torch.diag(cov_mar)
        
        return mean, std

    def evaluate(self, states, actions):
        """
            Estimates the values of each state and their corresponding log 
            probabilities.

            Return:
                log_probability: The log probability for the set of actions.
                entropy: //TODO
                entropy: The entropy of the policy distribution, which is to be used for
                    exploration.
        """
        mean, std = self.forward(states)
        distribution = Normal(mean, std)

        # Calculate the log probability for that action.
        log_probability = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)

        return log_probability, entropy