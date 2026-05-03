import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
from pathlib import Path
import numpy as np
import csv

from rl_PolicyNetwork import PolicyNetwork
from rl_ValueNetwork import ValueNetwork
from math import sqrt

class RLResourceAllocator:
    def __init__(self, state_space_dimensions, action_space_dimensions) -> None:
        self.state_space_dimensions = state_space_dimensions
        self.action_space_dimensions = action_space_dimensions

        # PPO Algorithm Step 1: Initialising the Policy (Actor) and Value (Critic) networks.
        self.policy_network = PolicyNetwork(self.state_space_dimensions, self.action_space_dimensions)
        self.value_network = ValueNetwork(self.state_space_dimensions)

        # Default Hyperparameter Values
        self.batch_size = 512           # Number of timesteps per episode.
        self.updates_per_episode = 5    # Number of times to update the policy/actor and value/critic networks per episode.
        self.learning_rate = 0.005      # Learning Rate of the policy and value optimisers.
        self.gamma = 0.95               # Discount factor to be used for calculating rewards-to-go.
        self.clip_parameter = 0.2       # Value to clip the ratio when calculating surrogate 2.
        self.entropy_coefficient = 0.01 # Value to multiply the entropy loss by to encourage/disencourage exploration. The value is the same as that used in Mahimalmur (2025).
        self.number_of_mini_batches = 8
        
        # Check the mini-batch size is valid.
        assert self.batch_size % self.number_of_mini_batches == 0, "The number of mini-batches must be a multiple of the batch size."


        # Optimisers for more stable convergence
        self.policy_optimiser = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimiser = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        # Set a seed value for reproducible results.
        #torch.manual_seed(1)

        self.batch_actions = []
        self.batch_states = []
        self.batch_log_probabilities = []
        self.batch_rewards = []

        # Try to load the Existing Policy and Value networks for future episodes.
        if Path("./ppo_policy.pth").is_file() and Path("./ppo_policy.pth").is_file():
            self.policy_network.load_state_dict(torch.load("ppo_policy.pth"))
            self.value_network.load_state_dict(torch.load("ppo_value.pth"))
            print("loaded .pth files")


    def get_action(self, state):
        """
            Returns the percentage of CPU that the resource allocator should give to the incoming task. This
            subroutine is to be called from OMNeT.

            Parameters:
                state: The current state of the simulation at the time a task is set to be allocated some CPU resources.
                The state space includes:
                    1. Required CPU cycles to process the input task.
                    2. Communication latency of that task.
                    3. Resource (CPU) Utilisation.
                    4. Queue length.
                    5. Total combined CPU cycles from the queue.     

            Returns:
                action (float): The percentage of CPU to be used to compute the input task given as a ratio.

                log_probability (float): The log probability of the action taken, which is just the confidence the network
                has of it being successful / maximising the reward.
        """
        state = torch.tensor(state, dtype=torch.float)

        # Query the Policy/Actor network for a mean action and the standard deviation.
        mean, std = self.policy_network(state)

        # Create a Gaussian distribution with the mean action and the standard deviation.
        distribution = Normal(mean, std)
        #print("Mean:", mean.mean().item(), "Std:", std.mean().item())

        # Sample an action from the distribution.
        raw_action = distribution.sample()

        # Calculate the log probability for that action to be successful.
        log_probability = distribution.log_prob(raw_action)

        return raw_action.detach(), log_probability.detach()

    def add_trajectory(self, action, log_probability, state, latency, energy_consumption, resource_utilisation):
        """
            Adds the input trajectory to the current batch.

            This subroutine is to be called from OMNeT upon completion of a timestep, 
            which for me is when a specific task has entered and finished being executed.

            Parameters:
                action (float)
                log_probability (float)
                state (array)
                latency (float)
        """     
        self.batch_actions.append(action)
        self.batch_log_probabilities.append(log_probability)
        self.batch_states.append(state)
        reward = self.compute_reward(latency, energy_consumption, resource_utilisation)
        self.batch_rewards.append(reward)
    
    def compute_reward(self, latency, energy_consumption, resource_utilisation):
        latency_weight = 0.7
        energy_consumption_weight = 0.3
        
        latency_baseline = 1000.0
        latency_reward = (latency_baseline - latency) / latency_baseline
        
        #latency_baseline2 = 2000        
        #latency_reward = -(latency/latency_baseline2)
        
        # TODO: Create a function that performs normalisation and takes two arguments.
        energy_baseline = 3
        energy_consumption_reward = (energy_baseline - energy_consumption) / energy_baseline
        
        # Weighted sum of the two reward components.
        total_reward = (latency_weight * latency_reward +
                        energy_consumption_weight * energy_consumption_reward)
        
        return latency_reward
        #return total_reward
    
    def learn(self):
        # PPO Algorithm Step 3: Collect trajectories/experiences from the most recent iteration/episode
        # and convert them into separate tensors.
        batch_actions = torch.tensor(self.batch_actions, dtype=torch.float)
        batch_states = torch.tensor(self.batch_states, dtype=torch.float)
        batch_log_probablities = torch.tensor(self.batch_log_probabilities, dtype=torch.float)
        batch_rewards = torch.tensor(self.batch_rewards, dtype=torch.float)

        # PPO Algorithm Step 4: Calculate Rewards to Go
        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        value = self.value_network(batch_states)

        # PPO Algorithm Step 5: Calculate Advantages
        advantages = batch_rewards_to_go - value.detach()

        # Normalise Advantages for improved stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        # Configure Mini-batches
        mini_batch_step = batch_actions.size(0)
        indices = np.arange(mini_batch_step)
        mini_batch_size = mini_batch_step // self.number_of_mini_batches
        print(mini_batch_size)

        for epoch in range(self.updates_per_episode):
                       
            # Randomly shuffle the indices for the mini-batch.
            for mini_batch_i in range(0, mini_batch_step, mini_batch_size):
                end = mini_batch_i + mini_batch_size
                index = indices[mini_batch_i:end]
                
                mini_batch_states = batch_states[index]
                mini_batch_actions = batch_actions[index]
                mini_batch_log_probabilities = batch_log_probablities[index]
                mini_batch_advantages = advantages[index]
                mini_batch_rewards_to_go = batch_rewards_to_go[index]            
            
                # Calculate the Value and Current Log probabilities for the current epoch.
                value = self.value_network(mini_batch_states)
                current_log_probabilities, entropy = self.policy_network.evaluate(mini_batch_states, mini_batch_actions)
    
                # Calculate ratios for the surrogate losses.
                ratios = torch.exp(current_log_probabilities - mini_batch_log_probabilities)
    
                # Calculate Surrogate Losses.
                surr1 = ratios * mini_batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_parameter, 1 + self.clip_parameter) * mini_batch_advantages   
    
                # PPO Algorithm Step 6: Update the Policy.
                policy_loss = (-torch.min(surr1, surr2)).mean()
                #policy_loss -= self.entropy_coefficient * entropy.mean()
    
                self.policy_optimiser.zero_grad()
                policy_loss.backward(retain_graph = True)
                self.policy_optimiser.step()
    
                # PPO Algorithm Step 7: Fit Value function by regression on MSE using the
                # predicted values at the current epoch.
                value_loss = nn.MSELoss()(value, mini_batch_rewards_to_go)
    
                self.value_optimiser.zero_grad()
                value_loss.backward()
                self.value_optimiser.step()
        
        average_reward = round(sum(self.batch_rewards)/len(self.batch_rewards), 4)
        min_reward = round(min(self.batch_rewards), 4)
        max_reward = round(max(self.batch_rewards), 4)
        average_advantage = round(advantages.mean().item(), 4)
        std_advantage = round(advantages.std().item(), 4)
        entropy_loss = round(entropy.mean().item(), 4)
        policy_loss = round(policy_loss.item(), 4)
        value_loss = round(value_loss.item(), 4)
        print("---- PPO Update Start ----")      
        print("Average reward:", average_reward)
        print("Reward min/max:", min_reward, max_reward)
        print("Advantage mean:", average_advantage)
        print("Advantage std:", std_advantage)
        print("Entropy Loss: ", entropy_loss)
        print("Policy loss:", policy_loss)
        print("Value loss:", value_loss)
        print("---- PPO Update Complete ----")
        
        row = [
            average_reward,
            min_reward,
            max_reward,
            average_advantage,
            std_advantage,
            entropy_loss,
            policy_loss,
            value_loss
            ]
        
        # Print to CSV for testing
        filename = "1_PPOTrainingData.csv"
        file_exists = Path(f"./{filename}").is_file()
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            print(file_exists)
            
            # Write the header if the file doesn't exist.
            if not file_exists:
                writer.writerow(["Average Reward","Reward Min","Reward Max", "Advantage Mean", "Advantage Std", "Entropy Loss", "Policy Loss", "Value Loss"])
            
            writer.writerow(row) 
  

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []

        discounted_reward = 0

        for reward in reversed(batch_rewards):
            discounted_reward = reward + discounted_reward * self.gamma

            batch_rewards_to_go.insert(0, discounted_reward)

        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        return batch_rewards_to_go


    def clear(self):
        self.batch_actions = []
        self.batch_log_probabilities = []
        self.batch_states = []
        self.batch_rewards = []


    def update_and_save(self):
        """
            A subroutine triggered whenever an episode ends.
        """
        # PPO Algorithm Step 2: Learn for some number of iterations. In this case,
        # an interation's length = to the batch size, which itself is equal to the episode length.


        # Update both the policy and value networks.
        self.learn()
        
        last_action = self.batch_actions[-1]        

        # Once the episode has ended, clear the batch to prepare for the new one.
        self.clear()
        
        print("---- Debugging ----")
        debug_state = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float)
        mean, std = self.policy_network(debug_state)
        print("DEBUG POLICY:", mean.item(), std.item())
        
        #print("log_std grad:", self.policy_network.log_std.grad)
        
        print("RAW ACTION: ", last_action)
        print("CLIPPED ACTION: ", min(1, max(0.01, last_action)))
            
        for name, param in self.policy_network.named_parameters():
            print("WEIGHT CHECK:", name, param.mean().item())
            if param.grad is not None:
                print(name, param.grad.abs().mean().item())

        torch.save(self.policy_network.state_dict(), "./ppo_policy.pth")
        torch.save(self.value_network.state_dict(), "./ppo_value.pth")