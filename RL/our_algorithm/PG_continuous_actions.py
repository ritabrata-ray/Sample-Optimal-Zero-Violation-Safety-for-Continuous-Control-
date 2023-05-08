import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from envs.car_yaw_dynamics_4D import car_dynamics



class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

car = car_dynamics()
input_size = car.get_state_dim()
hidden_size = 100
output_size = car.get_action_dim()
lr = 0.001
GAMMA = 0.99

policy = Policy(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy.parameters(), lr=lr)


def update_policy(policy, optimizer, rewards, log_probs):
    returns = []
    for t in range(len(rewards)):
        Gt = 0
        discount = 1
        for r in rewards[t:]:
            Gt = Gt + discount * r
            discount = discount * GAMMA
        returns.append(Gt)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    policy_loss = []
    for log_prob, Gt in zip(log_probs, returns):
        policy_loss.append(-log_prob * Gt)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


n_episodes = 2000
max_timesteps = 1000

reward_returns_across_traning_episodes = []
cost_returns_across_training_episodes = []


for i_episode in range(n_episodes):
    car.reset()
    rewards = []
    log_probs = []
    costs = []
    actions = []
    DONE = False
    step_counter = 0
    INFO = " "
    while not DONE:
        state = car.get_state()
        state = torch.tensor(state)
        action_mean = policy(state)
        action_std = torch.tensor([0.4]) #0.1 for mountain car with action clamped between +-1
        action_dist = Normal(action_mean, action_std)
        action = action_dist.sample()
        action = torch.clamp(action, -10.0, 10.0)
        log_prob = action_dist.log_prob(action)
        log_probs.append(log_prob)
        step_counter += 1
        action = action.detach().numpy()
        action = action[0]
        reward, cost, DONE, INFO = car.step(action)
        rewards.append(reward)
        costs.append(cost)
        actions.append(action)



    update_policy(policy, optimizer, rewards, log_probs)
    episode_reward_returns = []
    episode_cost_returns = []
    for t in range(len(rewards)):
        Gt = 0
        Ct = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        episode_reward_returns.append(Gt)
        pw = 0
        for c in costs[t:]:
            Ct = Ct + GAMMA ** pw * c
            pw = pw + 1
        episode_cost_returns.append(Ct)
    print("INFO = {} in episode: {} after {} steps.".format(INFO, i_episode+1, step_counter))
    print("Actions like: {}".format(actions[step_counter-1]))
    reward_returns_across_traning_episodes.append(episode_reward_returns[0])
    cost_returns_across_training_episodes.append(episode_cost_returns[0])

    if i_episode % 10 == 0:
        print('Episode {}\tReward: {:.2f}'.format(i_episode, sum(rewards)))

episodes = np.arange(n_episodes)

plt.plot(episodes,reward_returns_across_traning_episodes, label="Sum Rewards for every episode")
plt.xlabel("Training episodes")
plt.ylabel("Discounted Returns for each episode/trajectory")
plt.title("Vanilla REINFORCE algorithm for car dynamics")
plt.legend(loc='lower right', borderpad=0.4, labelspacing=0.7)
#plt.savefig(os.path.join(file_path,"Bandits_Comparison.pdf"), format="pdf", bbox_inches="tight")
plt.show()

plt.plot(episodes,cost_returns_across_training_episodes, label="Sum Costs for every episode")
plt.xlabel("Training episodes")
plt.ylabel("Discounted Costs for each episode/trajectory")
plt.title("Vanilla REINFORCE algorithm for car dynamics")
plt.legend(loc='lower right', borderpad=0.4, labelspacing=0.7)
#plt.savefig(os.path.join(file_path,"Bandits_Comparison.pdf"), format="pdf", bbox_inches="tight")
plt.show()