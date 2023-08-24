import torch
import numpy as np
import random


def time_mask(signal, Pt = 0.5, num_masks=1, T=16, replace_with_zero=False):
    cloned = signal.clone()
    Sl = cloned.shape[0]
    v = torch.rand(1)
    if v < Pt:  
        for i in range(0, num_masks):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, Sl - t)

            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (replace_with_zero):
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned
    else:
        return cloned

def noise_augment(signal, Pn=0.5):
  cloned = torch.clone(signal)
  v = torch.rand(1)
  if v < Pn:
    return cloned + torch.randn(1)
  else: 
    return cloned

def gaussian_noise(signal, P=0.5):
  cloned = torch.clone(signal).squeeze(1)
  v = torch.rand(1)
  if v < P:
    noise = torch.FloatTensor(cloned.shape[0]).uniform_(-np.abs(cloned.mean()/3), np.abs(cloned.mean()/3))
    cloned = cloned + noise
    return cloned.unsqueeze(1)
  else:
    return cloned.unsqueeze(1)
