#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################


import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, p_drop=.95):
        super(DQN, self).__init__()

        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # create layers
        self.network = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.network.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], self.action_size)

        # dropout
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x):
        """Build a network that maps state -> action values."""

        # define the layers
        for linear in self.network:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)

