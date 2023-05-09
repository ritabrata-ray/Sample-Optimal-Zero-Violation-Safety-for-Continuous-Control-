import torch
import torch.nn as nn
import torch.optim as optim
import gym
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=256):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_space, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_space),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_space, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

def ppo_lagrangian(env_name, epochs=10, batch_size=64, gamma=0.99, clip_ratio=0.2, lr=3e-5, betas=(0.9, 0.999), render=True):
    # Initialize the environment and the agent
    env = gym.make(env_name)
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    agent = ActorCritic(obs_space, action_space)
    optimizer = optim.Adam(agent.parameters(), lr=lr, betas=betas)

    # Store the rewards and costs for every episode
    rewards_list = []
    costs_list = []

    # Run the training loop
    for epoch in range(epochs):
        obs_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        log_prob_batch = []
        cost_batch = []

        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        while not done:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            policy, value = agent(obs)
            action = torch.normal(policy.squeeze(0), 0.1).detach().numpy()
            obs_batch.append(deepcopy(obs))
            action_batch.append(deepcopy(action))
            obs, reward, done, _, _ = env.step(action)
            cost = reward + 1
            episode_reward += reward
            episode_cost += cost
            reward_batch.append(reward)
            value_batch.append(value)
            log_prob = -0.5 * (torch.tensor(action) - policy.squeeze(0)).pow(2).sum() / (2 * 0.1**2) - 0.5 * np.log(2 * np.pi * 0.1**2) * torch.tensor(action_space)
            log_prob_batch.append(log_prob.clone().detach())
            cost_batch.append(cost)

        rewards_list.append(episode_reward)
        costs_list.append(episode_cost)
        # Compute the advantage estimate
        returns = []
        advantages = []
        R = 0
        for r in reward_batch[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        for value, R in zip(value_batch, returns):
            advantage = R - value.item()
            advantages.append(advantage)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        total_loss = 0
        # Compute the surrogate objective function
        
        optimizer.zero_grad()
        for i in range(min(batch_size, len(obs_batch))):
            old_policy = agent.actor(obs_batch[i])
            old_log_prob = torch.log(old_policy.squeeze(0)[action_batch[i]])
            ratio = torch.exp(log_prob_batch[i] - old_log_prob)
            clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            surrogate1 = clip * advantages[i]
            surrogate2 = ratio * advantages[i]
            loss = -torch.min(surrogate1, surrogate2)

            # Apply the Lagrangian penalty term to the loss
            eps = 0.2
            constraint = torch.max(torch.abs(policy - old_policy)).item()
            penalty = -0.5 * eps * constraint ** 2
            loss = loss - penalty
            total_loss += loss.item()

            loss.backward()
        optimizer.step()

        # Compute and print the total rewards and costs for the episode
        episode_reward = sum(reward_batch)
        episode_cost = -penalty.item()

        # Add the rewards and costs to the respective lists
        rewards_list.append(episode_reward)
        costs_list.append(episode_cost)

        # Print the episode number, total loss, and total reward for the episode
        print(f"Epoch: {epoch} Loss: {total_loss/batch_size:.4e} Reward: {episode_reward:.2f} Cost: {episode_cost:.2f}")
    env.close()

    # Plot the rewards and costs for every episode
    # plt.plot(rewards_list)
    # plt.title("Sum Rewards for every episode")
    # plt.xlabel("Episode")
    # plt.ylabel("Sum Rewards")
    # plt.show()

    # plt.plot(costs_list)
    # plt.title("Sum Costs for every episode")
    # plt.xlabel("Episode")
    # plt.ylabel("Sum Costs")
    # plt.show()    

env_name = 'MountainCarContinuous-v0'

def main():
    ppo_lagrangian(env_name, epochs=50, batch_size=64, gamma=0.99, clip_ratio=0.2, lr=3e-4, betas=(0.9, 0.999), render=False)

if __name__ == '__main__':
    main()