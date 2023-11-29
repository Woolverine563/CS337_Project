import numpy as np
import torch

def one_hot_encoding(x):
    X = [int(elem) for elem in x]
    total_num = np.max(X) + 1
    return np.eyes(total_num)[X]

def jitter(sample, jitter_ratio):
    return sample + np.random.normal(loc = 0, scale = jitter_ratio, size = sample.shape)

def DataTransform_TD(sample, jitter_ratio):
    return jitter(sample, jitter_ratio)

def rem_frq(sample, perturb_ratio):
    return sample * ((torch.FloatTensor(sample.shape).uniform_() > perturb_ratio).to(sample.device))

def add_frq(sample, perturb_ratio):
    mask = ((torch.FloatTensor(sample.shape).uniform_() > 1 - perturb_ratio).to(sample.device))
    perturb = torch.rand(mask.shape)*(sample.max()*0.1)
    return sample + mask*perturb

def DataTransform_FD(sample, perturb_ratio = 0.1):
    return rem_frq(sample, perturb_ratio) + add_frq(sample, perturb_ratio)


