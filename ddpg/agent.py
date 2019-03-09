#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this declaration at the top                              #
############################################################


import numpy as np
import torch
import torch.nn.functional as F
from model.base import BaseConfiguration, BaseAgent
from utils.funcs import make_int, make_tensor, to_np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent(BaseAgent, BaseConfiguration):
    def __init__(self, params):
        super(DDPGAgent, self).__init__(params)
        self.__name__ = "DDPG"
        self.max_steps = params["max_steps"]
        self.total_steps = 0
        self.update_steps = 0


    def act(self, states):
        config = self.config
        self.network.eval()
        with torch.no_grad():
            actions = self.network(states)
        self.network.train()
        actions = to_np(actions)
        actions += self.random_process.sample()
        actions = self.task.step(actions)
        return actions


    def step(self, states, actions, next_states, rewards, dones): #need to add others_states, others_actions, others_next_states to work with multi-agent
        config = self.config
        next_states = self.state_normalizer(next_states)
        self.episode_reward += np.mean(rewards)
        if np.any(dones):
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.random_process.reset_states()

        if self.update_steps <= 10:
            self.replay.add(states, actions, rewards, next_states, make_int(dones))  #need to add others_states, others_actions, others_next_states to work with multi-agent

        if 10 < self.update_steps <= 20:
            for _ in range(self.update_rate):
                if self.replay.__len__() >= config["memory_size"]:
                    experiences = self.replay.sample()
                    self.learn(experiences)

        self.total_steps += 1
        self.update_steps += 1

        if self.update_steps == 20:
            self.update_steps = 0


    def learn(self, experiences):
        config = self.config

        if config["multi_agent"]:
            states, actions, rewards, next_states, terminals, others_states, others_actions, \
            others_next_states = experiences
            states = torch.cat((states, others_states), dim=1).to(device)
            actions = torch.cat((actions.float(), others_actions.float()), dim=1).to(device)
            next_states = torch.cat((next_states, others_next_states), dim=1).to(device)
        else:
            states, actions, rewards, next_states, terminals, sample_indxs, weights = experiences
            states = states.squeeze(1)
            actions = actions.squeeze(1)
            next_states = next_states.squeeze(1)

        terminals = make_tensor(terminals)

        #critic optimization
        if config["multi_agent"]:
            phi_next_list = [self.target_network.feature(s) for s in [next_states, others_next_states]]
            a_next_list = [self.target_network.actor(phi) for phi in phi_next_list]
            a_next = torch.cat(a_next_list, dim=1).to(device)
            q_next = self.target_network.critic(phi_next_list, a_next)
        else:
            phi_next = self.target_network.feature(next_states)
            a_next = self.target_network.actor(phi_next)
            q_next = self.target_network.critic(phi_next, a_next)

        q_next = config["discount"] * q_next * (1 - terminals)
        q_next.add_(rewards)
        q_next = q_next.detach()
        phi = self.network.feature(states)
        q = self.network.critic(phi, make_tensor(actions))

        critic_loss = F.smooth_l1_loss(q, q_next)
        self.network.critic_opt.zero_grad()
        critic_loss.backward()
        if config["gradient_clip"]:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), config["gradient_clip"])
        self.network.critic_opt.step()

        #actor optimization
        action = self.network.actor(phi)
        if config["multi_agent"]:
            others_action = self.actor_local(others_states)
            others_action = others_action.detach()
            action = torch.cat((action, others_action), dim=1).to(device)
        policy_loss = -self.network.critic(phi.detach(), action)
        priorities = policy_loss + 1e-5
        policy_loss = policy_loss.mean()

        self.network.actor_opt.zero_grad()
        policy_loss.backward()
        if config["priority"]:
            self.memory.update_priorities(sample_indxs, priorities.data.cpu().numpy())  # update priorities in buffer
        self.network.actor_opt.step()

        self.soft_update(self.target_network.network, self.network.network, config["gae_tau"])


    def reset(self):
        self.random_process.reset_states()

