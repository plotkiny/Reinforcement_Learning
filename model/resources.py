#!usr/bin/env python

############################################################
# Copyright (C) 2020 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################


import numpy as np


class RescaleNormalizer():
    def __init__(self, coef=1.0, states=False):
        self.coef = coef
        self.states = states

    def __call__(self, x):
        if not self.states:
            x = np.asarray(x)
            return self.coef * x
        else:
            mean = x.mean(0)
            std = x.std(0) + 1.0e-10
            return mean, std


class Schedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1, type="linear", zeta=0.5):
        val = self.current
        if type == "variable":
            self.current  = np.mean(self.current + 0.5 * (np.random.random(20) - zeta))
        else:
            self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class OrnsteinUhlenbeckProcess():
    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(self.dt) * np.random.standard_normal(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

