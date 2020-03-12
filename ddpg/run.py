#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################


import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from utils.funcs import  load_yaml
from ddpg.ddpg import DeterministicActorCriticNet, FCBody, FCBodyWithAction
from ddpg.agent import DDPGAgent
from model.base import BaseTask
from model.resources import OrnsteinUhlenbeckProcess, Schedule, RescaleNormalizer
from utils.replay_buffer import PriorityReplayBuffer
from unityagents import UnityEnvironment


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Task(BaseTask):
    def __init__(self, env, train, num_agents=1):
        super(Task, self).__init__(env, train, num_agents)

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        return actions


def main(env, config, train=False):

    params = config["parameters"]
    params["task_fn"] = lambda: Task(env, train)
    params["state_normalizer"] =  RescaleNormalizer()
    params["batch_norm"] = RescaleNormalizer(states=True)

    action_dim = params["task_fn"]().action_dim
    state_dim = params["task_fn"]().state_dim

    params["network_fn"] = lambda: DeterministicActorCriticNet(state_dim, action_dim,
        actor_body=FCBody(state_dim, tuple(params["hidden_layers"]), gate=F.relu),
        critic_body=FCBodyWithAction(state_dim, action_dim, tuple(params["hidden_layers"]), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    params["replay_fn"] = lambda: PriorityReplayBuffer(action_size=action_dim,  # params.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
                                                    buffer_size=int(params["buffer_size"]),
                                                    batch_size=int(params["mini_batch_size"]),
                                                    seed=params["seed"])

    params["random_process_fn"] = lambda: OrnsteinUhlenbeckProcess(
        size=(params["task_fn"]().action_dim,), std=Schedule(0.2))

    agent = DDPGAgent(params)
    brain_name = env.brain_names[0]

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, agent.n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        states = agent.state_normalizer(states)
        scores = np.zeros(num_agents)  # initialize the score
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, next_states, rewards, dones)
            scores += rewards  # update the score
            states = next_states
            if np.any(dones):  # exit loop if episode finished
                break
            if agent.max_steps and agent.total_steps >= agent.max_steps:
                break

        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    torch.save(agent.network.state_dict(), 'checkpoint.pth')

    return scores, scores_window


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Command line arguments ")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--path", required=True, help="path to environment app")
    parser.add_argument("--train", action="store_true", help="train the model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = load_yaml(f)

    env = UnityEnvironment(file_name=args.path)
    main(env,config, args.train)

