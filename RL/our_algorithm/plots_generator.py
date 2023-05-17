import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
data_path = "/Users/ritabrataray/Desktop/neurips_2023/code/RL/our_algorithm/simulation_data"
episodes = np.arange(500)
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

srl_epi_lengths = np.zeros((10,500)) #10 experiments with 500 episodes each
srl_epi_lengths[0] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_1.npy'))
srl_epi_lengths[1] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_2.npy'))
srl_epi_lengths[2] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_3.npy'))
srl_epi_lengths[3] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_4.npy'))
srl_epi_lengths[4] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_5.npy'))
srl_epi_lengths[5] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_6.npy'))
srl_epi_lengths[6] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_7.npy'))
srl_epi_lengths[7] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_8.npy'))
srl_epi_lengths[8] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_9.npy'))
srl_epi_lengths[9] = np.load(os.path.join(data_path,'Safe_RL_episode_lengths_10.npy'))

srl_mean, srl_upper_bound, srl_lower_bound = bounding_plots(srl_epi_lengths)

usrl_epi_lengths = np.zeros((10,500)) #10 experiments with 500 episodes each
usrl_epi_lengths[0] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_1.npy'))
usrl_epi_lengths[1] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_2.npy'))
usrl_epi_lengths[2] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_3.npy'))
usrl_epi_lengths[3] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_4.npy'))
usrl_epi_lengths[4] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_5.npy'))
usrl_epi_lengths[5] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_6.npy'))
usrl_epi_lengths[6] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_7.npy'))
usrl_epi_lengths[7] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_8.npy'))
usrl_epi_lengths[8] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_9.npy'))
usrl_epi_lengths[9] = np.load(os.path.join(data_path,'UnSafe_RL_episode_lengths_10.npy'))

usrl_mean, usrl_upper_bound, usrl_lower_bound = bounding_plots(usrl_epi_lengths)


CPO_epi_lengths = np.zeros((10,500)) #10 experiments with 500 episodes each
CPO_epi_lengths[0] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_1.npy'))
CPO_epi_lengths[1] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_2.npy'))
CPO_epi_lengths[2] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_3.npy'))
CPO_epi_lengths[3] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_4.npy'))
CPO_epi_lengths[4] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_5.npy'))
CPO_epi_lengths[5] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_6.npy'))
CPO_epi_lengths[6] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_7.npy'))
CPO_epi_lengths[7] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_8.npy'))
CPO_epi_lengths[8] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_9.npy'))
CPO_epi_lengths[9] = np.load(os.path.join(data_path,'CPO_RL_episode_lengths_10.npy'))

CPO_mean, CPO_upper_bound, CPO_lower_bound = bounding_plots(CPO_epi_lengths)
# Now loading average safety rates

srl_safety_rates = np.zeros((10,500)) #10 experiments with 500 episodes each
srl_safety_rates[0] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_1.npy'))
srl_safety_rates[1] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_2.npy'))
srl_safety_rates[2] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_3.npy'))
srl_safety_rates[3] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_4.npy'))
srl_safety_rates[4] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_5.npy'))
srl_safety_rates[5] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_6.npy'))
srl_safety_rates[6] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_7.npy'))
srl_safety_rates[7] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_8.npy'))
srl_safety_rates[8] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_9.npy'))
srl_safety_rates[9] = np.load(os.path.join(data_path,'Safe_RL_safety_rate_10.npy'))

srl_mean_sr, srl_upper_bound_sr, srl_lower_bound_sr = bounding_plots(srl_safety_rates)

usrl_safety_rates = np.zeros((10,500)) #10 experiments with 500 episodes each
usrl_safety_rates[0] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_1.npy'))
usrl_safety_rates[1] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_2.npy'))
usrl_safety_rates[2] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_3.npy'))
usrl_safety_rates[3] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_4.npy'))
usrl_safety_rates[4] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_5.npy'))
usrl_safety_rates[5] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_6.npy'))
usrl_safety_rates[6] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_7.npy'))
usrl_safety_rates[7] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_8.npy'))
usrl_safety_rates[8] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_9.npy'))
usrl_safety_rates[9] = np.load(os.path.join(data_path,'UnSafe_RL_safety_rate_10.npy'))

usrl_mean_sr, usrl_upper_bound_sr, usrl_lower_bound_sr = bounding_plots(usrl_safety_rates)


CPO_safety_rates = np.zeros((10,500)) #10 experiments with 500 episodes each
CPO_safety_rates[0] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_1.npy'))
CPO_safety_rates[1] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_2.npy'))
CPO_safety_rates[2] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_3.npy'))
CPO_safety_rates[3] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_4.npy'))
CPO_safety_rates[4] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_5.npy'))
CPO_safety_rates[5] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_6.npy'))
CPO_safety_rates[6] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_7.npy'))
CPO_safety_rates[7] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_8.npy'))
CPO_safety_rates[8] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_9.npy'))
CPO_safety_rates[9] = np.load(os.path.join(data_path,'CPO_RL_safety_rate_10.npy'))

CPO_mean_sr, CPO_upper_bound_sr, CPO_lower_bound_sr = bounding_plots(CPO_safety_rates)


plot_path = os.path.join(data_path,"Plots")

plt.plot(episodes,srl_mean,label='Our Algorithm',color='red')
plt.fill_between(episodes,srl_lower_bound,srl_upper_bound,color='red',alpha=0.2)
plt.plot(episodes,usrl_mean, label='REINFORCE',color='green')
plt.fill_between(episodes,usrl_lower_bound,usrl_upper_bound,color='green',alpha=0.2)
plt.plot(episodes,CPO_mean, label='CPO',color='blue')
plt.fill_between(episodes,CPO_lower_bound,CPO_upper_bound,color='blue',alpha=0.2)
plt.xlabel("Training Episode")
plt.ylabel("Number of steps to complete the task")
#plt.title("Convergence of model-free safe RL algorithms")
plt.legend(loc='upper right', borderpad=0.4, labelspacing=0.7)
plt.savefig(os.path.join(plot_path,"Combined_Convergence_Plots.png"),format="png", bbox_inches="tight")
plt.show()

# Now safety rate bar plots
unsafe_srl = 0
unsafe_usrl = 0
unsafe_CPO = 0
for t in range(len(episodes)):
    unsafe_srl += (1 - srl_mean_sr[t])
    unsafe_usrl += (1 - usrl_mean_sr[t])
    unsafe_CPO += (1 - CPO_mean_sr[t])
unsafe_srl = unsafe_srl/(len(episodes))
unsafe_usrl = unsafe_usrl/(len(episodes))
unsafe_CPO = unsafe_CPO/(len(episodes))

algorithms=['REINFORCE', 'CPO', 'Our Algorithm']
mean_safety_rates={'REINFORCE':unsafe_usrl, 'CPO': unsafe_CPO, 'Our Algorithm': unsafe_srl}
algorithms=list(mean_safety_rates.keys())
safety_rates=list(mean_safety_rates.values())
XX=pd.Series(safety_rates,index=algorithms) #new code for broken y-axis bar graph
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
fig.text(0.02, 0.5, r"Fraction of episode unsafe", va="center", rotation="vertical", fontsize=12) #puts the ylabel on center even with broken axis
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom=False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim(0,0.1)
ax1.set_ylim(0.4,1.0)
ax1.set_yticks(np.arange(0.50,1.01,0.08))
XX.plot(ax=ax1,kind='bar',color=['green','blue','red'])
XX.plot(ax=ax2,kind='bar',color=['green','blue','red'])
for tick in ax2.get_xticklabels():
    tick.set_rotation(0)
d = .015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
plt.xlabel("Model-Free RL Algorithm")
#plt.ylabel("Fraction of episode unsafe") 3 broken y-axis shifted the ylabel before the axis break
#plt.title("Safety Rates of Different Algorithms")
plt.savefig(os.path.join(plot_path,"Bar_Safety_Rate_Plot.png"),format="png", bbox_inches="tight")
plt.show()

