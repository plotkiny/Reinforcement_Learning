#!usr/bin/env python

############################################################
# Copyright (C) 2020 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from dqn import DQN
from utils.replay_buffer import PriorityReplayBuffer

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
        self.soft_update(self.qnetwork_target, self.qnetwork_local, self.tau)

    def soft_update(self, target_model, local_model, tau):

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

