import matplotlib.pyplot as plt
import numpy as np
import os

data_path = "/Users/ritabrataray/Desktop/neurips_2023/code/RL/our_algorithm/simulation_data"

def mean_std_calculator(x):
    m = 0
    std = 0
    for i in range(len(x)):
        m += x[i]
    m /= len(x)
    for j in range(len(x)):
        std += (x[j] - m)**2
    std /= len(x)
    std = np.sqrt(std)

    return m, std

def bounding_plots(y):
    a , b = np.shape(y)
    empty_zero = np.zeros(a)
    mean_data = np.zeros(b)
    mean_plus_std = np.zeros(b)
    mean_minus_std = np.zeros(b)
    for i in range(b):
        for j in range(a):
            empty_zero[j] = y[j][i]
        mean_i, std_i = mean_std_calculator(empty_zero)
        mean_data[i] = mean_i
        mean_plus_std[i] = mean_i + std_i
        mean_minus_std[i] = mean_i - std_i
    return mean_data, mean_plus_std, mean_minus_std

data_epi_lengths = np.zeros((10,500)) #10 experiments with 500 episodes each
data_epi_lengths[0] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_1.npy'))
data_epi_lengths[1] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_2.npy'))
data_epi_lengths[2] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_3.npy'))
data_epi_lengths[3] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_4.npy'))
data_epi_lengths[4] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_5.npy'))
data_epi_lengths[5] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_6.npy'))
data_epi_lengths[6] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_7.npy'))
data_epi_lengths[7] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_8.npy'))
data_epi_lengths[8] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_9.npy'))
data_epi_lengths[9] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_10.npy'))

mean, upper_bound, lower_bound = bounding_plots(data_epi_lengths)
episodes = np.arange(500)

plt.plot(episodes, mean, label = "mean")
plt.plot(episodes, upper_bound, label ="upper_bound")
plt.plot(episodes, lower_bound, label ="lower_bound")
plt.show()

