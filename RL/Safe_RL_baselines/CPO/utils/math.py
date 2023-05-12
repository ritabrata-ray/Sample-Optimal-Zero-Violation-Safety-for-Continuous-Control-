import torch
import math
import pdb

#higher std to lower entropy and opposite
#entropy - avg surprisal(particular event from random variable) to probability distributions
def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

#continuous log distribution, expected value/mean and std of variable's natural logarithm
def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    #pdb.set_trace()
    return log_density.sum(1, keepdim=True)
