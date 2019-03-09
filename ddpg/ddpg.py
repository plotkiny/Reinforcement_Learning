#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this declaration at the top                              #
############################################################


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.funcs import  make_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def layer_init(layer, w_scale=1.0):
    nn.init.kaiming_normal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class DummyBody(nn.Module):

    #borrowed from https://github.com/ShangtongZhang/DeepRL

    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(400,300), gate=F.relu, p_drop=1):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(l1, l2)) for l1, l2 in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        for indx, layer in enumerate(self.layers):
            x = self.gate(layer(x))
            x = self.dropout(x)
        return x

class FCBodyWithAction(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(400, 300), gate=F.relu, agents=1):
        super(FCBodyWithAction, self).__init__()
        dims = copy.deepcopy(hidden_units)
        self.layers = nn.ModuleList([layer_init(nn.Linear((state_dim+action_dim) * agents, dims[0]))])
        self.layers.extend(nn.ModuleList([layer_init(nn.Linear(l1, l2)) for l1, l2 in zip(dims[:-1], dims[1:])]))
        self.gate = gate
        self.feature_dim = dims[-1]
        self.dims_length = len(dims)


    def forward(self, x, action):
        for indx, layer in enumerate(self.layers):
            if indx == 0:
                x = layer(torch.cat([x, action], dim=1))
            else:
                x = layer(x)
            x = self.gate(x)
        return x


class ActorCriticNet(nn.Module):

    # borrowed from https://github.com/ShangtongZhang/DeepRL

    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 3e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())



class DeterministicActorCriticNet(nn.Module):

    # borrowed from https://github.com/ShangtongZhang/DeepRL

    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(device)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = make_tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

