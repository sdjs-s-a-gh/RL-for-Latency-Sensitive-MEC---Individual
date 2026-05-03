import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal 
from torch.optim import Adam

import numpy as np
import time
from pathlib import Path
import csv

from rl_FeedForwardNN import FeedForwardNeuralNetwork
from rl_PolicyNetwork import PolicyNetwork

class PPO:
    def __init__(self, state_dimensions: int, action_dimension: int):
        # Collect information from the environment
        self.state_dimensions = state_dimensions
        self.action_dimensions = action_dimension

        # PPO Algorithm Step 1: Initialising the Policy (Actor) and Value (Critic) networks.
        self.policy_network = PolicyNetwork(self.state_dimensions, self.action_dimensions)
        self.value_network = FeedForwardNeuralNetwork(self.state_dimensions, 1)

        self._set_hyperparameters()
        self._load_networks()

        self.policy_optimiser = Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_optimiser = Adam(self.value_network.parameters(), lr=self.learning_rate)

        self.logger = {
            "delta_t": time.time_ns(),
            "buffer_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "average_advantage": 0,
            "std_advantage": 0,
            "entropy_losses": []
        }

        # Instantiate the buffers for data to be stored.
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_log_probabilities = []
        self.buffer_rewards = []

    def _set_hyperparameters(self) -> None:
        """
            Set the default hyperparameters for the PPO agent.
        """
        torch.manual_seed(298)
        self.updates_per_episode: int = 5
        
        self.gamma: float = 0.99
        self.clip_threshold = 0.2      # The clip threshold used in the surrogate losses.
        self.learning_rate = 0.00025      
        self.mini_batch_size = 64
        self.entropy_coefficient = 0.001
        
        #print(f"mini-batch size: {self.mini_batch_size}; entropy coefficient: {self.entropy_coefficient}")
        #print(f"updates: {self.updates_per_iteration}; lr: {self.learning_rate}; gamma: {self.gamma}; clip: {self.clip_threshold}")

    def _load_networks(self) -> None:
        """
            Loads the Policy and Value networks from respective
            .pth files should they exist.
        """

        # Try to load the Existing Policy and Value networks for future episodes.
        if Path("./ppo_policy.pth").is_file() and Path("./ppo_policy.pth").is_file():
            self.policy_network.load_state_dict(torch.load("ppo_policy.pth"))
            self.value_network.load_state_dict(torch.load("ppo_value.pth"))
            print("Successfully loaded .pth files")
        else:
            print("No neural network could be located and loaded into the program -"
            " or maybe one exists and the other does not. Regardless, the training" \
            " on this episode will start from scratch instead.")

    def get_action(self, state) -> tuple:
        #print(f"State: {state}")
        # Query the policy network for a mean action.
        # Create the Multivariate Normal distribution for using an action space > 1.
        distribution = self.policy_network.get_distribution(state)

        # Sample an action and its log probability from the distribution.
        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        # Remove the computation graphs from the action and log probability when returning.
        return action.detach().numpy(), log_probability.detach()

    def learn(self) -> None:      
        # PPO Algorithm Step 3: Collect trajectories/experiences from the episode just computed.
        # Firstly, convert each component from the buffer data into tensors for training.
        buffer_states = torch.tensor(np.array(self.buffer_states), dtype=torch.float)
        buffer_actions = torch.tensor(np.array(self.buffer_actions), dtype=torch.float)
        buffer_log_probabilities = torch.tensor(np.array(self.buffer_log_probabilities), dtype=torch.float)

        self.logger["buffer_rewards"].append(self.buffer_rewards)
        #print(self.buffer_rewards)

        buffer_rewards_to_go = self.compute_rewards_to_go(self.buffer_rewards)

        buffer_rewards_to_go = torch.tensor(np.array(buffer_rewards_to_go), dtype=torch.float)

        #batch_states, batch_actions, batch_log_probabilities, batch_rewards_to_go, batch_episode_lengths = self.rollout()

        # Query the value/critic network for a value V for each state in the buffer.
        value = self.value_network(buffer_states).squeeze()

        # PPO Algorithm Step 5: Calculate Advantages
        advantages = buffer_rewards_to_go - value.detach()

        # Normalise Advantages for improved stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        # Create mini-batches for to introduce more noise and avoid overfitting.
        batch_size = buffer_states.size(0)
        indices = np.arange(batch_size)
        
        for epoch in range(self.updates_per_episode):
            # Anneal the learning rate to improve convergence to avoid being either too quick or too slow.
            #fraction_completed: float | int = (timesteps_so_far - 1.0) / total_timesteps    # -1.0 is used as the timesteps have +1 added in the rollout() function.
            #print(f"Fraction Completed: {fraction_completed}; Timesteps so far: {timesteps_so_far}")
            # TODO: Also, I don't think this would be possible for me as the programme wouldn't track overall timesteps.
            #new_learning_rate = self.learning_rate * (1.0 - fraction_completed)
            #new_learning_rate = max(new_learning_rate, 0.0)
            #self.policy_optimiser.param_groups[0]["lr"] = new_learning_rate
            #self.value_optimiser.param_groups[0]["lr"] = new_learning_rate

            # Randomly shuffle the trajectories for the mini-batch
            np.random.shuffle(indices)
            for mini_batch in range(0, batch_size, self.mini_batch_size):
                end = mini_batch + self.mini_batch_size
                index = indices[mini_batch:end]

                mini_batch_states = buffer_states[index]
                mini_batch_actions = buffer_actions[index]
                mini_batch_log_probabilities = buffer_log_probabilities[index]
                mini_batch_advantages = advantages[index]
                mini_batch_rewards_to_go = buffer_rewards_to_go[index] 

                # Calculate current log probabilities (pi_theta(a_t | s_t)) for this current epoch.
                # The old log probabilities are stored in the variable batch_log_probabilities.
                #print(f"Batch States Size: {mini_batch_states.size()}; Batch Actions Size: {mini_batch_actions.size()}")
                #raise ValueError(f"Batch States Size: {mini_batch_states.sum(-1).size()}; Batch Actions Size: {mini_batch_actions.size()}")
                current_log_probabilities, entropy_loss = self.evaluate(mini_batch_states, mini_batch_actions)

                # Calculate the Value and Current Log probabilities for the current epoch.
                # TODO: Probably move this value into the same evaluate function as the current log probabilities uses.
                value = self.value_network(mini_batch_states).squeeze()

                # Calculate the ratios for the surrogate losses.
                ratios = torch.exp(current_log_probabilities - mini_batch_log_probabilities)

                # Calculate the surrogate losses.
                surr1 = ratios * mini_batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_threshold, 1 + self.clip_threshold) * mini_batch_advantages

                # Negate the policy loss in compliance with the Adam Optimiser, which maximises loss.
                policy_loss = (-torch.min(surr1, surr2)).mean()
                policy_loss = policy_loss - self.entropy_coefficient * entropy_loss

                # PPO Algorithm Step 6: Update the Policy.
                self.policy_optimiser.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_optimiser.step()

                # PPO Algorithm Step 7: Fit Value function by regression on MSE using the
                # predicted values at the current epoch.
                value_loss = nn.MSELoss()(value.squeeze(), mini_batch_rewards_to_go)

                self.value_optimiser.zero_grad()
                value_loss.backward()
                self.value_optimiser.step()

                self.logger["policy_losses"].append(policy_loss.detach())
                self.logger["value_losses"].append(value_loss.detach())
                self.logger["entropy_losses"].append(entropy_loss)

        self.logger["average_advantage"] = round(advantages.mean().item(), 4)
        self.logger["std_advantage"] = round(advantages.std().item(), 4)
        self._log_summary()

    def add_trajectory(self, state: list, raw_action: float | int, log_probability: float | int, latency: float | int, energy_consumption, queue_utilisation) -> None:
        """
            Collects a trajectory for a single timestep in the OMNeT++
            MEC environment and appends the data to the buffer. The
            reward for this timestep is subsequently performed on the
            Python-side code for a more appropriate separation of concerns. 
        """
        self.buffer_states.append(state)
        self.buffer_actions.append([raw_action])
        self.buffer_log_probabilities.append(log_probability)

        # Compute the reward for the given timestep.
        reward: float | int = self.compute_reward(latency, energy_consumption, queue_utilisation)

        self.buffer_rewards.append(reward)

    def compute_reward(self, latency: float | int, energy_consumption, queue_utilisation) -> float | int:
        """
            Calculates the reward for the outcome of a given timestep.
        """
        
        latency_baseline = 1000.0
        reward = (latency_baseline - latency) / latency_baseline

        return reward

    def compute_rewards_to_go(self, buffer_rewards: list[float | int]) -> torch.Tensor:
        buffer_rewards_to_go = []
        
        discounted_reward = 0

        for reward in reversed(buffer_rewards):
            discounted_reward = reward + discounted_reward * self.gamma
            buffer_rewards_to_go.insert(0, discounted_reward)
            
        # Convert the rewards-to-go into a tensor
        buffer_rewards_to_go = torch.tensor(buffer_rewards_to_go, dtype=torch.float)

        return buffer_rewards_to_go
    
    def evaluate(self, batch_states, batch_actions) -> tuple:
        # Calculate the most recent log probabilities of the batched actions using the
        # most recent actor network.
        #print(f"Batch States Size: {batch_states.size()}; Batch Actions Size: {batch_actions.size()}")
        #raise ValueError(f"Batch States Size: {batch_states.size()}; Batch Actions Size: {batch_actions.size()}")
        distribution = self.policy_network.get_distribution(batch_states)
        #print(f"Batch States Size: {batch_states.size()}; Batch Actions Size: {batch_actions.size()}")
        log_probability = distribution.log_prob(batch_actions)

        entropy = distribution.entropy().mean()

        return log_probability, entropy

    def update_and_save(self) -> None:
        """
            Triggers a function to train the PPO agent over the data stored in the
            buffer. Afterwards, the neural networks are then saved to respective
            .pth files to be used later. This subroutine is to be called once an
            episode has finished in the OMNeT environment.
        """
        # PPO Algorithm Step 2: Learn for some number of iterations.
        self.learn()

        # Save the neural networks.
        torch.save(self.policy_network.state_dict(), "./ppo_policy.pth")
        torch.save(self.value_network.state_dict(), "./ppo_value.pth")

    def _log_summary(self) -> None:
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        average_episode_rewards = np.mean([self.logger["buffer_rewards"]])
        minimum_episode_rewards = np.min(self.logger["buffer_rewards"])
        maximum_episode_rewards = np.max(self.logger["buffer_rewards"])

        average_policy_loss = np.mean([losses.float().mean() for losses in self.logger["policy_losses"]])
        average_value_loss = np.mean([losses.float().mean() for losses in self.logger["value_losses"]])
        average_entropy_loss = np.mean([losses.detach().float().mean() for losses in self.logger["entropy_losses"]])

        # Round to x number of decimal places for more intuitive and intepretable logging messages.
        average_episode_rewards = str(round(average_episode_rewards, 5))
        minimum_episode_rewards = str(round(minimum_episode_rewards, 5))
        maximum_episode_rewards = str(round(maximum_episode_rewards, 5))
        average_policy_loss = str(round(average_policy_loss, 5))
        average_value_loss = str(round(average_value_loss, 5))
        last_policy_loss = str(round(self.logger["policy_losses"][-1].item(), 5))
        last_value_loss = str(round(self.logger["value_losses"][-1].item(), 5))
        average_entropy_loss = str(round(average_entropy_loss, 4))
        last_entropy_loss = str(round(self.logger["entropy_losses"][-1].item(), 5))       


        print(flush=True)
        print(f"---- Iteration # ----")
        print(f"Average Episodic Rewards: {average_episode_rewards}")   # Average Rewards is same as mine
        print(f"Min Reward: {minimum_episode_rewards}")                             # Min/Max rewards is the same as mine
        print(f"Max Reward: {maximum_episode_rewards}")
        # print(f"Average Rewards-to-Go: {average_rewards_to_go}")
        # print(f"Min Rewards-to-Go: {min_rewards_to_go}")
        # print(f"Max Rewards-to-Go: {max_rewards_to_go}")
        print(f"Average Advantage: {self.logger["average_advantage"]}")             # Avg/std advantages is the same as mine
        print(f"Std Advantage: {self.logger["std_advantage"]}")
        print(f"Average Entropy Loss: {average_entropy_loss}")
        print(f"Last Entropy Loss: {last_entropy_loss}")
        print(f"Average Policy Loss: {average_policy_loss}")
        print(f"Last Policy Loss: {last_policy_loss}")              # Last policy and value losses should be the same as mine
        print(f"Average Value Loss: {average_value_loss}")
        print(f"Last Value Loss: {last_value_loss}")
        print(f"Iteration took: {delta_t} secs")
        print(f"---- Iteration Over ----")
        print(flush=True)

        filename = "1_OptimisedPPOTrainingData.csv"
        file_exists = Path(f"./{filename}").is_file()

        row: list = [
            average_episode_rewards,
            minimum_episode_rewards,
            maximum_episode_rewards,
            self.logger["average_advantage"],
            self.logger["std_advantage"],
            average_entropy_loss,
            last_entropy_loss,
            average_policy_loss,
            last_policy_loss,
            average_value_loss,
            last_value_loss,
        ]
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            
            # Write the header if the file doesn't exist.
            if not file_exists:
                writer.writerow([
                    "Average Reward",
                    "Min Reward",
                    "Max Reward",
                    "Advantage Mean",
                    "Advantage Std",
                    "Average Entropy Loss", 
                    "Last Entropy Loss",
                    "Average Policy Loss",
                    "Last Policy Loss", 
                    "Average Value Loss",
                    "Last Value Loss", 
                ])
            
            writer.writerow(row)
