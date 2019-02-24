#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from navigation_model import DQN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    """Interacts with and learns from the environment."""

    def __init__(self, params, state_size, action_size, seed, hidden_layers):
        super(Agent, self).__init__()
        
        """
        Params
        ======
            config: configuration file
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = int(params["batch_size"])
        self.beta_frames = int(params["beta_frames"])
        self.buffer_size = int(params["buffer_size"])
        self.gamma = params["gamma"]
        self.lr = params["lr"]
        self.p_alpha = params["p_alpha"]
        self.p_beta = params["p_beta"]
        self.tau = params["tau"]
        self.update_every = params["update_every"]

        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size, seed, hidden_layers=hidden_layers).to(device)
        self.qnetwork_target = DQN(state_size, action_size, seed, hidden_layers=hidden_layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = PriorityReplayBuffer(action_size, self.buffer_size, self.batch_size, seed, self.p_alpha,
                                           self.p_beta, self.beta_frames)
        self.t_step = 0 # Initialize time step (for updating every UPDATE_EVERY steps)

    def step(self, state, action, reward, next_state, done, i_episode):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, i_episode, self.gamma)

    def act(self, state, i_episode=0.0, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        max_action_indx = np.argmax(action_values.cpu().data.numpy())
        if i_episode == 0:
            epsilon = 1.0 / (i_episode + 1.0)
        else:
            epsilon = 1.0 / i_episode

        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.action_size) * epsilon / self.action_size
        policy_s[max_action_indx] = 1 - epsilon + (epsilon / self.action_size)

        if random.random() > eps:
            return int(policy_s[max_action_indx])
        else:
            return int(np.random.choice(np.arange(self.action_size), p=policy_s))

    def adjust_learning_rate(self, lr, optimizer, i_episode):
        """Sets the learning rate to the initial LR decayed by 10 every 50 episodes"""
        lr = lr * (0.0001 ** (i_episode // 40))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def learn(self, experiences, i_episode, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sample_indxs, weights  = experiences

        #get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        #get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        #compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        #adjust learning rate at the beginning
        if i_episode < 40:
            lr = self.get_lr(self.optimizer)
            self.adjust_learning_rate(lr, self.optimizer, i_episode)

        #compute loss
        loss = F.mse_loss(Q_expected, Q_targets, reduce=False)
        priorities = loss + 1e-5
        loss = loss.mean()

        #minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(sample_indxs, priorities.data.cpu().numpy())  #update priorities in buffer
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):

        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



class PriorityReplayBuffer:
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
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

