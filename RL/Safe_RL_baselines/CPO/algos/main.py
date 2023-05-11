import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from contenv.RL.envs.car_yaw_dynamics_4D import *

from utils import *
import statistics as st
from models.continuous_policy import Policy
from models.critic import Value
from models.discrete_policy import DiscretePolicy
from algos.cpo import cpo_step
from core.common import estimate_advantages, estimate_constraint_value
from core.agent import Agent
import pdb
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
CUDA_LAUNCH_BLOCKING=1

#summarizing using tensorboard
from torch.utils.tensorboard import SummaryWriter

# Parse arguments 
args = parse_all_arguments()

dtype = torch.float64
torch.set_default_dtype(dtype)
#track gpu and tensors
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)

"""environment"""
#commented out printing
env = car_dynamics()
#state dimensions - screenshot of the game, characteristics described as a vector
#mountain car - velocity and position, midpoint is origin
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()
#print(state_dim)
#action space shape is numerical structure of the legitimate actions that can b applied
#use discrete policy if is_disc_action
is_disc_action = True #len(env.action_space.shape) == 0
#print(env.action_space.shape)
#mean, std, etc from the observation state
running_state = ZFilter((state_dim,), clip=5)


"""seeding - changed due to gym seeding changes"""
np.random.seed(args.seed)
#env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#env.seed(args.seed)

"""create all the paths to save learned models/data"""
save_info_obj = save_info(assets_dir(), args.exp_num, args.exp_name, args.env_name) #model saving object
save_info_obj.create_all_paths() # create all paths
writer = SummaryWriter(os.path.join(assets_dir(), save_info_obj.saving_path, 'runs/')) #tensorboard summary
 
"""define actor and critic"""
if args.model_path is None:
    #if not using pre-trained model, choose discrete policy
    #should usually be discrete policy
    #discrete vs continuous - pre-defined actions or learned actions/spaced out data vs constant data
    if False and is_disc_action:
        policy_net = DiscretePolicy(state_dim, action_dim)
    else:
        policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

"""create agent"""
#learner and decision maker, give environment, policy, running state
agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)

#cost of 0.1 - rewards kept going down???
#rewards that worked: 0.01, 0.03
"""define constraint cost function"""    
#cost function: 0 if within certain area and direction, cost otherwise

def constraint_cost(state, action, c):
    costs = tensor(0.0 * np.ones(len(state)), dtype=dtype)
    for i in range(len(state)):
        costs[i]=c[i]
    costs.to(device)
    return costs

def update_params(batch, d_k=0):
    states = torch.from_numpy(np.stack(batch.state)[:args.max_batch_size]).to(dtype).to(device) #[:args.batch_size]
    actions = torch.from_numpy(np.stack(batch.action)[:args.max_batch_size]).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)[:args.max_batch_size]).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)[:args.max_batch_size]).to(dtype).to(device)
    costs = torch.from_numpy(np.stack(batch.cost)[:args.max_batch_size]).to(dtype).to(device)
        
    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    #take information and estimate the advantages
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
    #c = constraint_cost(states, actions, costs)
    #print(c)
    cost_advantages, _ = estimate_advantages(costs, masks, values, args.gamma, args.tau, device)
    constraint_value = estimate_constraint_value(costs, masks, args.gamma, device)
    constraint_value = constraint_value[0]

    """perform update"""
    #update the policy
    v_loss, p_loss, cost_loss = cpo_step(args.env_name, policy_net, value_net, states, actions, returns, advantages, cost_advantages, constraint_value, d_k, args.max_kl, args.damping, args.l2_reg)
    return v_loss, p_loss, cost_loss


def main_loop():
    # variables and lists for recording losses
    v_loss = 0
    p_loss = 0
    cost_loss = 0
    v_loss_list = []
    p_loss_list = []
    cost_loss_list = []
    
    # lists for dumping plotting data for agent
    rewards_std = []
    env_avg_reward = []
    num_of_steps = []
    num_of_episodes = []
    total_num_episodes = []
    total_num_steps = []
    tne = 0 #cummulative number of episodes
    tns = 0 #cummulative number of steps
    
    # lists for dumping plotting data for mean agent
    eval_avg_reward = []
    eval_avg_reward_std = []
    
    if args.algo_name == "CPO":
        # define initial d_k
        #bound the constraint
        d_k = args.max_constraint
        # define annealing factor
        if args.anneal == True:
            e_k = args.annealing_factor
        else:
            e_k = 0            
    
    # for saving the best model
    best_avg_reward = 0
    if args.env_name == "CartPole-v1" or args.env_name == "MountainCar-v0":
        best_std = 20
    else:
        best_std = 5
    
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        #collect samples of env and update the parameters with it
        t0 = time.time()
        v_loss, p_loss, cost_loss = update_params(batch, d_k)
        t1 = time.time()
        
        d_k = d_k + d_k*e_k  # max constraint annealing 
        
        # update lists for saving
        v_loss_list.append(v_loss)
        p_loss_list.append(p_loss)
        cost_loss_list.append(cost_loss)
        rewards_std.append(log['std_reward']) 
        env_avg_reward.append(log['env_avg_reward'])
        num_of_steps.append(log['num_steps'])
        num_of_episodes.append(log['num_episodes'])
        tne = tne + log['num_episodes']
        tns = tns + log['num_steps']
        total_num_episodes.append(tne)
        total_num_steps.append(tns)
        
        # update tensorboard summaries
        writer.add_scalars('losses', {'v_loss':v_loss}, i_iter)
        writer.add_scalars('losses', {'p_loss':p_loss}, i_iter)
        writer.add_scalars('losses', {'cost_loss':cost_loss}, i_iter)
        writer.add_scalars('data/rewards', {'env_avg_reward':log['env_avg_reward']}, i_iter) #env_avg_reward
        writer.add_scalar('env_avg_reward', log['env_avg_reward'], i_iter)
        writer.add_scalar('num of episodes', log['num_episodes'], i_iter)
        writer.add_scalar('num of steps', log['num_steps'], i_iter)

        # evaluate the current policy
        running_state.fix = True  #Fix the running state
        agent.num_threads = 20
        agent.mean_action = False
        '''
        if args.env_name == "CartPole-v1" or args.env_name == "MountainCar-v0":
            agent.mean_action = False
        else:
            agent.mean_action = True
        '''
        seed = np.random.randint(1,1000)
        #env.action_space.seed(seed)
        eval_reward_type = 4
        #collect samples for the agent and evaluate the policy
        _, eval_log = agent.collect_samples(20000)
        running_state.fix = False
        agent.num_threads = args.num_threads
        agent.mean_action = False
        
        # update eval lists
        eval_avg_reward.append(eval_log['env_avg_reward'])
        eval_avg_reward_std.append(eval_log['std_reward'])
            
        # print learning data on screen     
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.2f}\tTest_R_avg {:.2f}\tTest_R_std {:.2f}'.format( i_iter, log['sample_time'], t1-t0, log['env_avg_reward'], eval_log['env_avg_reward'], eval_log['std_reward']))              
        
        # save the best model
        if eval_log['env_avg_reward'] >= best_avg_reward and eval_log['std_reward'] <= best_std:
            print('Saving new best model !!!!')
            to_device(torch.device('cpu'), policy_net, value_net)
            save_info_obj.save_models(policy_net, value_net, running_state)
            to_device(device, policy_net, value_net)
            best_avg_reward = eval_log['env_avg_reward']
            best_std = eval_log['std_reward']
            iter_for_best_avg_reward = i_iter+1
        
        # save some intermediate models to sample trajectories from
        if args.save_intermediate_model > 0 and (i_iter+1) % args.save_intermediate_model == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            save_info_obj.save_intermediate_models(policy_net, value_net, running_state, i_iter)
            to_device(device, policy_net, value_net)
        
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        
    # dump expert_avg_reward, num_of_steps, num_of_episodes
    save_info_obj.dump_lists(best_avg_reward, num_of_steps, num_of_episodes, total_num_episodes, total_num_steps, rewards_std, env_avg_reward, v_loss_list, p_loss_list, eval_avg_reward, eval_avg_reward_std)

# To detach each tensor in the list
    #for i in range(len(env_avg_reward)):
        #env_avg_reward[i] = env_avg_reward[i].detach().numpy()
    
    plt.plot(env_avg_reward)
    plt.show()

    print('Best eval R:', best_avg_reward)
    return best_avg_reward, best_std
if __name__ == '__main__':
    main_loop()
