#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################

import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.
        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done",
                                                  "others_states",
                                                  "others_actions",
                                                  "others_next_states"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, others_states,
            others_actions, others_next_states):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done,
                            others_states, others_actions, others_next_states)
        self.memory.append(e)

    def sample(self):
        '''Randomly sample a batch of experiences from memory.'''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences
                                  if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences
                                   if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences
                                   if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state
                                                  for e in experiences
                                                  if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences
                                            if e is not None]).astype(np.uint8)).float().to(device)

        others_states = torch.from_numpy(np.vstack([e.others_states for e in experiences
                                  if e is not None])).float().to(device)
        others_actions = torch.from_numpy(np.vstack([e.others_actions for e in experiences
                                   if e is not None])).float().to(device)
        others_next_states = torch.from_numpy(np.vstack([e.others_next_states
                                                      for e in experiences
                                                      if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, others_states,
                others_actions, others_next_states)

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)


class PriorityReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, p_alpha=0.75, p_beta=0.5, beta_frames=10000):

        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            p_alpha: sampling hyperparameter
            p_beta: update hyperparameter
            beta_frames: frames for updating beta

        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.beta_frames = beta_frames
        self.buffer_size = buffer_size
        self.current_frame = 1
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", ])
        self.memory = deque(maxlen=buffer_size)
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        self.position = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):

        assert(int(state.ndim) == int(next_state.ndim))

        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)

        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.position] = e

        #queue priorities
        max_priority = self.priorities.max() if self.memory else 1.0
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.buffer_size

    def beta_by_frame(self, indx):
        return min(1.0, self.p_beta + indx * (1.0 - self.p_beta) / self.beta_frames)

    def sample(self):

        if len(self.memory) == self.buffer_size:
            prior = self.priorities
        else:
            prior = self.priorities[:self.position]

        #sampling probability
        sample_prob = prior ** self.p_alpha
        sample_prob /= sample_prob.sum()

        #randomly sample a batch of experiences from memory.
        sample_indxs = np.random.choice(np.arange(len(self.memory)), self.batch_size, p=sample_prob)
        experiences = [self.memory[i] for i in sample_indxs]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(
            np.uint8)).float().to(device)

        #adaptive beta for annealing the amount of importance
        current_beta = self.beta_by_frame(self.current_frame)
        self.current_frame += 1

        #len of buffer
        total = len(self.memory)

        #remove noise, calculate minimum for all ALL probabilities, not just sampled
        prob_min = sample_prob.min()
        weight_max = (prob_min * total) ** (-current_beta)

        weights = (total * sample_prob[sample_indxs]) ** -current_beta
        weights /= weight_max.max()
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return (states, actions, rewards, next_states, dones, sample_indxs, weights)

    def update_priorities(self, batch_indxs, batch_priorities):
        for indx, priority in zip(batch_indxs, batch_priorities):
            self.priorities[indx] = (priority + 1e-5) ** self.p_alpha

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

