import torch
import torch.nn as nn
import torch.optim as optim
import gym
from copy import deepcopy
import matplotlib.pyplot as plt

# torch.autograd.set_detect_anomaly(True)

class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=256):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_space, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_space),
            nn.Softmax(dim=-1)
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
    env = gym.make(env_name,render_mode="human")
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = ActorCritic(obs_space, action_space)
    optimizer = optim.Adam(agent.parameters(), lr=lr, betas=betas)

    # Initialize lists to record rewards and losses
    total_rewards = []
    total_losses = []

    # Run the training loop
    for epoch in range(epochs):
        obs_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        log_prob_batch = []

        obs = env.reset()
        done = False
        while True:
            obs = torch.FloatTensor(obs).unsqueeze(0)
            policy, value = agent(obs)
            action = torch.multinomial(policy, 1).item()
            log_prob = torch.log(policy.squeeze(0)[action])

            obs_batch.append(deepcopy(obs))
            action_batch.append(action)
            obs, reward, done, truncate = env.step(action)

            reward_batch.append(reward)
            value_batch.append(value)
            log_prob_batch.append(log_prob)
            if done or truncate:
                break

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
        for i in range(min(batch_size,len(obs_batch))):
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
        # total_loss.backward()
        optimizer.step()

        # Record the total reward and loss for the epoch
        total_rewards.append(sum(reward_batch))
        total_losses.append(total_loss / batch_size)
        
        print(f"Epoch: {epoch} Loss: {total_loss/batch_size:.4e} rewards: {sum(reward_batch)}")

    env.close()
    
    # Plot the rewards and losses
    plt.plot(total_rewards)
    plt.title("Total Rewards per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.show()

#Write Main function here
env_name = 'CartPole-v1'

def main():
    ppo_lagrangian(env_name, epochs=50, batch_size=64, gamma=0.99, clip_ratio=0.2, lr=3e-4, betas=(0.9, 0.999), render=True)

if __name__ == '__main__':
    main()