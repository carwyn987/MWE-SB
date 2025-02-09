import torch
from torch import nn
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from gymnasium.wrappers import ClipAction
import random

from plotter.save_returns import gen_filepth, save_returns
from plotter.env_normalizers import pendulum_v1_normalizer

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

env_name = "Pendulum-v1"
gravity = 4.0
env = gym.make(env_name, render_mode=None, g=gravity)
prior_observation, info = env.reset()
n_obs = env.observation_space.shape[0]

n_hidden = 32
n_actions = env.action_space.shape[0]
print("Size of actions: ", n_actions, ", Size of observations: ", n_obs)

behavior_q_network = GenericNetwork(n_obs + 1, n_hidden, 1)
target_q_network = GenericNetwork(n_obs + 1, n_hidden, 1)
policy_network = GenericNetwork(n_obs, n_hidden, n_actions)

lr=1e-3
behavior_q_optimizer = torch.optim.Adam(behavior_q_network.parameters(), lr=lr)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)

replay_buffer = deque(maxlen=10000)

epsilon = 0.1
gamma = 0.99
num_minibatch_train = 100
synchronize_num_steps = 1000
num_episodes = 2500
return_save = []
qvalue_loss_save = []
policy_loss_save = []
n_step = 0

for ep in tqdm(range(num_episodes)):
    episode_over = False
    _return = 0
    while not episode_over:
        action = policy_network(torch.tensor(prior_observation)).item() if np.random.random() > epsilon else env.action_space.sample().item()
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, reward, terminated, truncated, info = env.step(clipped_action)
        episode_over = terminated or truncated
        replay_buffer.append((prior_observation, clipped_action, reward, observation, episode_over))
        prior_observation = observation
        _return += reward
        n_step += 1

        if n_step % synchronize_num_steps == 0:
            target_q_network.load_state_dict(behavior_q_network.state_dict())
        
    prior_observation, info = env.reset()
    return_save.append(_return)

    if len(replay_buffer) > num_minibatch_train:

        sampled_batch =random.sample(replay_buffer, k=num_minibatch_train)
        sample_state, sample_act, sample_reward, sample_next_state, sample_done = zip(*sampled_batch)
        prior_observation_tensor = torch.tensor(np.stack(sample_state), dtype=torch.float32)
        observation_tensor = torch.tensor(np.stack(sample_next_state), dtype=torch.float32)
        reward_tensor = torch.tensor(np.stack([sample_reward]), dtype=torch.float32)
        action_tensor = torch.tensor(np.stack(sample_act), dtype=torch.long)
        done_indexes = torch.tensor(np.stack([sample_done]).astype(int).T)

        #with torch.no_grad():
        next_actions = policy_network(observation_tensor).clamp(env.action_space.low.item(), env.action_space.high.item())  # Shape: [batch, n_actions=1]
        q_target = reward_tensor.T + gamma * target_q_network(torch.cat((observation_tensor, next_actions), dim=1)) * (1-done_indexes)

        # Update Critic (Action Value Network)
        behavior_q_optimizer.zero_grad()
        current_q_values = behavior_q_network(torch.cat((prior_observation_tensor, action_tensor), dim=1)) # should this action tensor be clipped?
        q_loss = nn.MSELoss()(current_q_values, q_target.float())
        q_loss.backward()
        behavior_q_optimizer.step()
        qvalue_loss_save.append(q_loss.item())

        # Update Actor (Policy Network)
        for param in behavior_q_network.parameters():
            param.requires_grad = False
        
        policy_optimizer.zero_grad()
        predicted_actions = policy_network(prior_observation_tensor)
        # Ensure actions are within valid range
        #predicted_actions = torch.clamp(predicted_actions, env.action_space.low[0], env.action_space.high[0])
        actor_q_values = behavior_q_network(torch.cat((prior_observation_tensor, predicted_actions), dim=1))
        policy_loss = -torch.mean(actor_q_values)
        policy_loss.backward()
        policy_optimizer.step()
        policy_loss_save.append(policy_loss.item())
        
        for param in behavior_q_network.parameters():
            param.requires_grad = True

# Save data with OPTIONAL plotter
try:
    normalized_return_save = pendulum_v1_normalizer(return_save)
    save_returns(returns=normalized_return_save, file_name=gen_filepth(model_name='ddpg', environment_name=env_name, additional_name='vanilla')) 
except Exception as err:
    print(f"Unexpected {err=} while saving returns, {type(err)=}")

# Show example behavior
env = gym.make(env_name, render_mode="human", g=gravity)
observation, info = env.reset()
episode_over = False
while not episode_over:
    action = policy_network(torch.tensor(observation)).item()
    action = np.clip(action, env.action_space.low, env.action_space.high)
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
env.close()

# Plot
plt.figure()
plt.title("Loss")
plt.plot(qvalue_loss_save, label="Q-value loss")
plt.plot(policy_loss_save, label="Policy loss")
plt.figure()
plt.title("Return")
plt.plot(return_save)
plt.show()
