#!usr/bin/env python

import numpy as np
import torch
import yaml
import yamlordereddictloader

from ddpg.run import device

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

def load_yaml(file_obj):
    return yaml.load(file_obj, Loader=yamlordereddictloader.Loader)

def to_np(t):
    return t.cpu().detach().numpy()

def make_int(a):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    return a.astype(np.uint8)

def make_tensor(x):
    if not isinstance(x, torch.FloatTensor):
        return torch.tensor(x, device=device, dtype=torch.float32)
    return x


