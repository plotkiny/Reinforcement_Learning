#!usr/bin/env python

############################################################
# Copyright (C) 2020 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################

import numpy as np
from ddpg.agent import DDPGAgent


class MultiAgent(object):

    def __init__(self, params, action_size, state_size, nb_agents):
        super(MultiAgent, self).__init__()

        '''Initialize an MultiAgent object.
        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        '''
        # Replay memory
        self.__name__ = "MADDPG"
        self.nb_agents = nb_agents
        self.na_idx = np.arange(self.nb_agents)
        self.action_size = action_size
        self.act_size = action_size * nb_agents
        self.state_size = state_size * nb_agents
        self.l_agents = [DDPGAgent(params) for i in range(nb_agents)]
        self.memory = params["replay_fn"]
        self.max_steps = params["max_steps"]
        self.total_steps = 0

    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(self.l_agents, states, actions, rewards, next_states,
                         dones)
        for i, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            na_filtered = self.na_idx[self.na_idx != i]
            others_states = states[na_filtered]
            others_actions = actions[na_filtered]
            others_next_states = next_states[na_filtered]
            agent.step(state, action, reward, next_state, done, others_states,
                       others_actions, others_next_states)
            if i==0:
                self.total_steps = agent.total_steps

    def act(self, states):
        na_rtn = np.zeros([self.nb_agents, self.action_size])
        for idx, agent in enumerate(self.l_agents):
            na_rtn[idx, :] = agent.act(states[idx])
        return na_rtn

    def reset(self):
        for agent in self.l_agents:
            agent.reset()

    def __len__(self):
        return self.nb_agents

    def __getitem__(self, key):
        return self.l_agents[key]


