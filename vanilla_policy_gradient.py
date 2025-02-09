# Vanilla Policy Gradient implementation - CartPole-v1
# Inspired by OpenAI Spinning Up: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
# ~ 90 lines of code (including imports, whitespace, evaluation)
# No "reward-to-go", baseline, discounted returns, or gpu training
# Author: Carwyn Collinsworth, 2025

import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from torch.distributions import Categorical

from plotter.save_returns import gen_filepth, save_returns

class PolicyGradientNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_inputs, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_outputs)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.softmax(self.fc2(x), dim=0)
        return x

env = gym.make("CartPole-v1", render_mode=None)
prior_observation, info = env.reset()

n_inputs = env.observation_space.shape[0]
n_hidden = 32
n_outputs = env.action_space.n
network = PolicyGradientNetwork(n_inputs, n_hidden, n_outputs)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
def loss_fn(data_store):
    loss = 0
    for (obs, act, _return) in data_store:
        log_probs = -1.0 * torch.log(torch.gather(network(obs), 1, act.unsqueeze(1)))
        loss += torch.sum(log_probs * _return)
    return loss/len(data_store)
    
    # Single trajectory training
    # obs, act, _return = data_store[-1]
    # log_probs = -1.0 * torch.log(torch.gather(network(obs), 1, act.unsqueeze(1)))
    # return torch.sum(log_probs * _return)

num_episodes = 2500
epsilon = 0.1
return_save = []
loss_save = []
data_store = deque(maxlen=10000)

for episode_idx in tqdm(range(num_episodes)):
    data_store.append([])
    episode_over = False
    while not episode_over:
        action_probs = network(torch.tensor(prior_observation))
        action = Categorical(action_probs).sample().item()
        observation, reward, terminated, truncated, info = env.step(action)
        data_store[-1].append((prior_observation, action, reward))
        prior_observation = observation
        episode_over = terminated or truncated
    observation, info = env.reset()

    _return = sum([reward for _, _, reward in data_store[-1]])
    obss, acts, _ = map(list, zip(*data_store[-1]))
    obss, acts = torch.tensor(np.stack(obss)), torch.tensor(np.array(acts))
    data_store[-1] = (obss, acts, _return)

    
    optimizer.zero_grad()
    loss = loss_fn(data_store)
    loss.backward()
    optimizer.step()
    return_save.append(_return)
    loss_save.append(loss.item())
env.close()

# Save data with OPTIONAL plotter
try:
    normalized_return_save = [x/env.spec.reward_threshold for x in return_save]
    save_returns(returns=normalized_return_save, file_name=gen_filepth(model_name='vpg', environment_name='cartpole-v1', additional_name='tuned-1')) 
except Exception as err:
    print(f"Unexpected {err=} while saving returns, {type(err)=}")

# Evaluation
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
episode_over = False
while not episode_over:
    action = torch.argmax(network(torch.tensor(observation))).item()
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
env.close()

plt.figure()
plt.title("Loss")
plt.plot(loss_save)
plt.figure()
plt.title("Return")
plt.plot(return_save)
plt.show()