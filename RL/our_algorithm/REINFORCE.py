import numpy as np
import math
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

from envs.car_yaw_dynamics_4D import car_dynamics



# Hyperparameters
GAMMA = 0.99
lr = 7e-3 # started with 3e-4 was too slow. lr=1e-2 was too fast. lr= 7e-3 seems right.


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    #pdb.set_trace()
    #return log_density.sum(1, keepdim=True)
    return log_density


#### Continuous action generating policy network (Actor Network) ####

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh', log_std=0, learning_rate = lr): #earlier log_std was 0, trying a more certain action
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        #self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        self.action_log_std = torch.tensor(log_std, dtype=torch.double)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        #action_log_std = self.action_log_std.expand_as(action_mean)
        action_log_std = self.action_log_std
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_log_prob(self, x, actions):
        #pdb.set_trace()
        action_mean, action_log_std, action_std = self.forward(x)
        #print(normal_log_density(actions, action_mean, action_log_std, action_std))
        return normal_log_density(actions, action_mean, action_log_std, action_std)



#### Policy update function ########

def update_policy(policy_network, critic_network, rewards, states, log_probs):
    discounted_advantages = []

    for t in range(len(rewards)):
        Gt = 0
        At = 0
        pw = 0
        state_t = states[t]
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
            At = Gt - critic_network(state_t)
        discounted_advantages.append(At)

    discounted_advantages = torch.tensor(discounted_advantages)
    discounted_advantages = (discounted_advantages - discounted_advantages.mean()) / (
                discounted_advantages.std() + 1e-2)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_advantages):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

##### Critic Network #####


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(100, 100), activation='tanh', learning_rate=lr): #earlier lr was 3e-4
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value

def update_critic(critic_network, rewards, states):
    returns = []
    for t in range(len(rewards)):
        Gt=0
        pw=0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        returns.append(Gt)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-2)  # normalize discounted rewards
    TD_loss_critic = []
    for state, Gt in zip(states ,returns):
        TD_loss_critic.append(torch.square( Gt- critic_network(state)))

    critic_network.optimizer.zero_grad()
    TD_loss_critic = torch.stack(TD_loss_critic).sum()
    TD_loss_critic.backward()
    critic_network.optimizer.step()



# Now we define the training loop



def train_actor_critic_REINFORCE(actor, critic, NUM_EPOCHS = 500):
    car = car_dynamics()  # instantiating car object
    episode_counter = 0
    reward_returns_across_traning_episodes = []
    cost_returns_across_training_episodes = []
    for epoch in range(NUM_EPOCHS):
        car.reset()
        rewards = []
        costs = []
        log_probs = []
        states = []
        actions = []
        DONE = False
        INFO = ""
        episode_counter += 1
        step_counter=0
        while not DONE:
            step_counter += 1
            state = car.get_state()
            #state = state.astype(np.float32) #This is being done inside the environment file.
            state = torch.tensor(state)
            states.append(state) #store tensor states
            action = actor.select_action(state)
            log_prob = actor.get_log_prob(state,action) #calculate it before stepping on the action
            car_action = action.detach().numpy()
            car_action = car_action[0] #need a floating point value instead of a 1-D array
            reward, cost, DONE, INFO = car.step(car_action)
            costs.append(cost)
            rewards.append(reward)
            actions.append(action) #store tensor actions
            log_probs.append(log_prob)
        update_policy(actor, critic, rewards, states, log_probs)
        update_critic(critic, rewards, states)
        # Now we store the sum_rewards and sum_costs (Discounted) for each episode
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
        print("INFO = {} in episode: {} after {} steps.".format(INFO,episode_counter, step_counter))
        reward_returns_across_traning_episodes.append(episode_reward_returns[0])
        cost_returns_across_training_episodes.append(episode_cost_returns[0])
    return reward_returns_across_traning_episodes, cost_returns_across_training_episodes

NUM_EPOCHS = 500 # make it 500 later
car = car_dynamics()
episodes = np.arange(NUM_EPOCHS) # will plot this in x-axis
actor = Policy(car.get_state_dim(), car.get_action_dim())
critic = Value(car.get_state_dim())

epi_r , epi_c = train_actor_critic_REINFORCE(actor, critic, NUM_EPOCHS)

#print(epi_r, epi_c)  # these can be plotted against "episodes" above but they should be averaged over 5 or 10 experiments and the plots need to be with mean
# and fill_between +_ 1 std_dev of the experiments.

plt.plot(episodes,epi_r, label="Sum Rewards for every episode")
#plt.plot(episodes, epi_c, label="Sum Costs for every episode")
plt.xlabel("Training episodes")
plt.ylabel("Discounted Returns for each episode/trajectory")
plt.title("Vanilla REINFORCE algorithm for car dynamics")
plt.legend(loc='lower right', borderpad=0.4, labelspacing=0.7)
#plt.savefig(os.path.join(file_path,"Bandits_Comparison.pdf"), format="pdf", bbox_inches="tight")
plt.show()

plt.plot(episodes,epi_c, label="Sum Costs for every episode")
plt.xlabel("Training episodes")
plt.ylabel("Discounted Costs for each episode/trajectory")
plt.title("Vanilla REINFORCE algorithm for car dynamics")
plt.legend(loc='lower right', borderpad=0.4, labelspacing=0.7)
#plt.savefig(os.path.join(file_path,"Bandits_Comparison.pdf"), format="pdf", bbox_inches="tight")
plt.show()

