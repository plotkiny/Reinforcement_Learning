#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this declaration at the top                              #
############################################################


import torch
from torch.optim import lr_scheduler
from utils.funcs import close_obj


class BaseConfiguration(object):
    def __init__(self, params):
        super(BaseConfiguration, self).__init__()
        self.config = params
        self.max_steps = int(params["max_steps"])
        self.buffer_size = int(params["buffer_size"]),
        self.batch_size = int(params["batch_size"])
        self.seed = torch.manual_seed(params["seed"])
        self.n_episodes = params["n_episodes"]
        self.memory_size = params["memory_size"]
        self.state_normalizer = params["state_normalizer"]
        self.batch_norm = params["batch_norm"]
        self.task = params["task_fn"]()
        self.network = params["network_fn"]()
        self.target_network = params["network_fn"]()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = params["replay_fn"]()
        self.random_process = params["random_process_fn"]()
        self.update_rate = params["update_every"]
        self.episode_reward = 0
        self.episode_rewards = []


class BaseAgent(object):
    def __init__(self, params):
        super(BaseAgent, self).__init__(params)
        self.config = params

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)

    def adjust_learning_rate(self, optim, type="step"):
        if type == "function":
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.95 ** epoch
            scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=[lambda1, lambda2])
        elif type == "step":
            scheduler = lr_scheduler.StepLR(optim, step_size=4, gamma=0.1)
        return scheduler

    def get_lr(self, optim):
        for param_group in optim.param_groups:
            return param_group["lr"]

    def soft_update(self, target, src, tau):

        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            src_model (PyTorch model): weights will be copied from (i.e. current)
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for target_param, src_param in zip(target, src):
            target_param.detach_()
            target_param.copy_(src_param * tau + target_param * (1.0 - tau))


class BaseTask(object):
    def __init__(self, env, train, num_agents):
        super(BaseTask, self).__init__()
        if 'Wrapper' in str(env):
            env = env.env
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.brain.vector_observation_space_size
        self.state_size *= self.brain.num_stacked_vector_observations
        self.num_agents = num_agents
        self.train = train

    def reset(self):
        pass

    def step(self, actions):
        pass

    def close(self):
        self.env.close()

