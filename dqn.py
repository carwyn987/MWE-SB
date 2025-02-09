"""
DQN is a Q-Learning-based RL algorithm for discrete action spaces
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
    
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#print(f"Using {device} device")

env = gym.make("CartPole-v1", render_mode=None)
prior_observation, info = env.reset()
state_space = env.observation_space.shape[0]
n_actions = env.action_space.n
print("|Observation| = ", state_space, ", |Action| = ", n_actions)

n_hidden = 32
lr = 1e-3
synchronize_num_steps = 100
behavior_q_model = GenericNetwork(state_space, n_hidden, n_actions)
target_q_model = GenericNetwork(state_space, n_hidden, n_actions)
action_value_optimizer = torch.optim.Adam(behavior_q_model.parameters(), lr=lr)
action_value_loss_fn = nn.MSELoss()

num_episodes = 2500
epsilon = 0.1
gamma = 0.99
num_train_iters = 1
num_minibatch_train = 50
return_save = []
behavior_q_model_loss_save = []
replay_buffer = deque(maxlen=10000)

n_step = 0
for episode_idx in tqdm(range(num_episodes)):
    episode_over = False
    _return = 0
    while not episode_over:
        action_values = behavior_q_model(torch.tensor(prior_observation))
        action = torch.argmax(action_values).item() if np.random.random() > epsilon else torch.randint(0, action_values.size(0), (1,)).item()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        replay_buffer.append((prior_observation, action, reward, observation, episode_over))
        _return += reward

        if len(replay_buffer) > num_minibatch_train:
            # Single sample method
            # for i in range(num_minibatch_train):
            #     (sample_state, sample_act, sample_reward, sample_next_state, sample_done) = random.choice(replay_buffer)

            #     prior_observation_tensor = torch.tensor(sample_state, dtype=torch.float32)
            #     observation_tensor = torch.tensor(sample_next_state, dtype=torch.float32)
            #     reward_tensor = torch.tensor(sample_reward, dtype=torch.float32)
            #     action_tensor = torch.tensor(sample_act, dtype=torch.long)
                
            #     # Q-value update
            #     with torch.no_grad():
            #         next_action_values = target_q_model(observation_tensor)
            #         target_q_value = reward_tensor + gamma * torch.max(next_action_values) * (1 - sample_done)
                
            #     predicted_q_value = behavior_q_model(prior_observation_tensor)[action_tensor]
            #     action_value_loss = action_value_loss_fn(predicted_q_value, target_q_value)
                
            #     action_value_optimizer.zero_grad()
            #     action_value_loss.backward()
            #     action_value_optimizer.step()
            #     behavior_q_model_loss_save.append(action_value_loss.item())

            # Batch method
            for i in range(num_train_iters):

                sampled_batch =random.sample(replay_buffer, k=num_minibatch_train)
                sample_state, sample_act, sample_reward, sample_next_state, sample_done = zip(*sampled_batch)
                prior_observation_tensor = torch.tensor(np.stack(sample_state), dtype=torch.float32)
                observation_tensor = torch.tensor(np.stack(sample_next_state), dtype=torch.float32)
                reward_tensor = torch.tensor(np.stack([sample_reward]), dtype=torch.float32)
                action_tensor = torch.tensor(np.stack([sample_act]), dtype=torch.long)
                done_indexes = np.stack([sample_done]).astype(int)
                
                # Q-value update
                with torch.no_grad():
                    next_action_values = target_q_model(observation_tensor)
                    max_vals, max_indxs = torch.max(next_action_values, dim=1)
                    target_q_value = reward_tensor.T + gamma * max_vals.unsqueeze(1) * (1 - done_indexes.T)
                    
                
                predicted_q_value = behavior_q_model(prior_observation_tensor).gather(1, action_tensor.T)
                action_value_loss = action_value_loss_fn(predicted_q_value, target_q_value.float())
                
                action_value_optimizer.zero_grad()
                action_value_loss.backward()
                action_value_optimizer.step()
                behavior_q_model_loss_save.append(action_value_loss.item())

        
        prior_observation = observation
        n_step += 1

        if n_step % synchronize_num_steps == 0:
            target_q_model.load_state_dict(behavior_q_model.state_dict())
    observation, info = env.reset()
    return_save.append(_return)
env.close()

# Save data with OPTIONAL plotter
try:
    normalized_return_save = [x/env.spec.reward_threshold for x in return_save]
    save_returns(returns=normalized_return_save, file_name=gen_filepth(model_name='dqn', environment_name='cartpole-v1', additional_name='tuned-v1')) 
except Exception as err:
    print(f"Unexpected {err=} while saving returns, {type(err)=}")

# Evaluation
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
episode_over = False
while not episode_over:
    with torch.no_grad():
        action = torch.argmax(behavior_q_model(torch.tensor(observation))).item()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
env.close()

# Plot
fig, (ax0, ax1) = plt.subplots(1,2)
ax0.set_title("Action Value Loss")
ax0.plot(behavior_q_model_loss_save)
ax1.set_title("Return")
ax1.plot(return_save)
plt.show()