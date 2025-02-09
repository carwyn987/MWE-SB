"""
Dyna-Q is a Q-Learning model-based RL algorithm for discrete action spaces
CartPole-v1 is a gymnasium environment with left and right actions
Gymnasium link: https://gymnasium.farama.org/environments/classic_control/cart_pole/
Author: Carwyn Collinsworth
Date: 02/05/25
"""

import torch
from torch import nn
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import random

from plotter.save_returns import gen_filepth, save_returns

class GenericNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
    def forward(self, input):
        logits = self.linear_relu_stack(input)
        return logits
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# Set up environment
env = gym.make("CartPole-v1")
prior_observation, info = env.reset()
state_space = env.observation_space.shape[0]
n_actions = env.action_space.n
print("|Observation| = ", state_space, ", |Action| = ", n_actions)

# Set up networks
n_hidden = 32
lr = 1e-3
reward_model = GenericNetwork(state_space, n_hidden, 1).to(device)
forward_model = GenericNetwork(state_space + 1, n_hidden, state_space).to(device)
action_value_model = GenericNetwork(state_space, n_hidden, n_actions).to(device)

# Set up optimizers
reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
forward_model_optimizer = torch.optim.Adam(forward_model.parameters(), lr=lr)
action_value_optimizer = torch.optim.Adam(action_value_model.parameters(), lr=lr)

# Set up loss functions
action_value_loss_fn = lambda x,y: (x - y)**2
reward_loss_fn = lambda x,y: (x - y)**2
forward_model_loss_fn = lambda x,y: torch.sum((x - y)**2)

num_episodes = 2500
train_iters = 1
epsilon = 0.1
gamma = 0.99
return_save = []
action_value_model_loss_save = []
reward_model_loss_save = []
forward_model_loss_save = []
replay_buffer = deque(maxlen=10000)

for episode_idx in tqdm(range(num_episodes)):
    episode_over = False
    _return = 0
    while not episode_over:
        # Interact with environment
        prior_observation_tensor = torch.tensor(prior_observation, dtype=torch.float32).to(device)
        action_values = action_value_model(prior_observation_tensor)
        action = torch.argmax(action_values).item() if np.random.random() > epsilon else random.choice(range(n_actions))
        observation, reward, terminated, truncated, info = env.step(action)
        replay_buffer.append((prior_observation, action))
        episode_over = terminated or truncated
        _return += reward

        # Tensorize quantities
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).reshape(1).to(device)
        action_tensor = torch.tensor([action], dtype=torch.float32).to(device)
        
        # Q-value update
        #with torch.no_grad():
        target_q_value = reward_tensor + gamma * action_value_model(observation_tensor).max().item() * (1 - episode_over)
        predicted_q_value = action_value_model(prior_observation_tensor)[action]
        action_value_loss = action_value_loss_fn(target_q_value, predicted_q_value)
        action_value_optimizer.zero_grad()
        action_value_loss.backward()
        action_value_optimizer.step()
        action_value_model_loss_save.append(action_value_loss.item())

        # Train forward-model one step
        concat_input = torch.concatenate((prior_observation_tensor, action_tensor.to(device)), dim=0)
        predicted_state = forward_model(concat_input)
        forward_model_loss = forward_model_loss_fn(predicted_state, observation_tensor)
        forward_model_optimizer.zero_grad()
        forward_model_loss.backward()
        forward_model_optimizer.step()
        forward_model_loss_save.append(forward_model_loss.item())

        # Train reward function one step
        predicted_reward = reward_model(prior_observation_tensor)
        reward_loss = reward_loss_fn(reward_tensor, predicted_reward)
        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()
        reward_model_loss_save.append(reward_loss.item())

        prior_observation = observation

        # Training loop with simulated experience
        for it in range(train_iters):
            (saved_prior_obs, saved_action) = random.choice(replay_buffer)
            saved_prior_obs_tensor = torch.tensor(saved_prior_obs, dtype=torch.float32).to(device)
            saved_action_tensor = torch.tensor([saved_action], dtype=torch.float32).to(device)
            
            # Simulate look-ahead
            saved_concat_input = torch.concatenate((saved_prior_obs_tensor, saved_action_tensor), dim=0)
            sim_obs_tensor = forward_model(saved_concat_input)
            sim_reward_tensor = reward_model(saved_prior_obs_tensor)
            target_q_value = sim_reward_tensor + gamma * torch.max(action_value_model(sim_obs_tensor))

            predicted_q_value = action_value_model(saved_prior_obs_tensor)[saved_action]
            action_value_loss = action_value_loss_fn(target_q_value, predicted_q_value)
            
            action_value_optimizer.zero_grad()
            action_value_loss.backward()
            action_value_optimizer.step()
            action_value_model_loss_save.append(action_value_loss.item())

    prior_observation, info = env.reset()
    return_save.append(_return)
env.close()

# Save data with OPTIONAL plotter
try:
    normalized_return_save = [x/env.spec.reward_threshold for x in return_save]
    save_returns(returns=normalized_return_save, file_name=gen_filepth(model_name='dyna-q', environment_name='cartpole-v1', additional_name='vanilla-v3')) 
except Exception as err:
    print(f"Unexpected {err=} while saving returns, {type(err)=}")

# Evaluation test run
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
episode_over = False
while not episode_over:
    with torch.no_grad():
        action = torch.argmax(action_value_model(torch.tensor(observation).to(device))).item()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
env.close()

# Plot return and losses
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4)
ax0.set_title("Return")
ax0.plot(return_save)
ax1.set_title("Action Value Loss")
ax1.plot(action_value_model_loss_save)
ax2.set_title("Reward Loss")
ax2.plot(reward_model_loss_save)
ax3.set_title("Forward Model Loss")
ax3.plot(forward_model_loss_save)
plt.show()