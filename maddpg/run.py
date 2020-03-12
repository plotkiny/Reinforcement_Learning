#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################


import argparse
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from collections import deque
from ddpg.ddpg import DeterministicActorCriticNet, FCBody, FCBodyWithAction
from maddpg.maddpg import MultiAgent
from model_.base import BaseTask
from model_.resources import OrnsteinUhlenbeckProcess, Schedule, RescaleNormalizer
from utils.funcs import load_yaml
from utils.replay_buffer import ReplayBuffer
from unityagents import UnityEnvironment


class Task(BaseTask):

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return states, rewards, dones


def main(env, config, train=False):

    params = config["parameters"]
    data_prefix = params["data_prefix"]
    n_episodes = params["n_episodes"]
    nb_agents = params["nb_agents"]
    solved = params["solved"]

    params["state_normalizer"] = RescaleNormalizer()
    params["batch_norm"] = RescaleNormalizer(states=True)
    params["task_fn"] = lambda: Task(env, train, )

    action_size = params["task_fn"]().action_size
    state_size = params["task_fn"]().state_size

    params["network_fn"] = lambda: DeterministicActorCriticNet(state_size, action_size,
                                                               actor_body=FCBody(state_size,
                                                                                 tuple(params["hidden_layers"]),
                                                                                 gate=F.relu),
                                                               critic_body=FCBodyWithAction(state_size, action_size,
                                                                                            tuple(params[
                                                                                                      "hidden_layers"]),
                                                                                            gate=F.relu),
                                                               actor_opt_fn=lambda params: torch.optim.Adam(params,
                                                                                                            lr=1e-4),
                                                               critic_opt_fn=lambda params: torch.optim.Adam(params,
                                                                                                             lr=1e-3))
    params["replay_fn"] = lambda: ReplayBuffer(action_size=action_size,
                                               buffer_size=int(params["buffer_size"]),
                                               batch_size=int(params["batch_size"]),
                                               seed=params["seed"])

    params["random_process_fn"] = lambda: OrnsteinUhlenbeckProcess(
        size=(params["task_fn"]().action_size,), std=Schedule(0.2))

    scores = []
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)
    agent = MultiAgent(params, action_size, state_size, nb_agents)

    print('\nTRAINING:')
    for episode in range(n_episodes):
        states = env.reset()
        score = np.zeros(len(agent))
        while True:
            actions = agent.act(states)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break
            if agent.max_steps and agent.total_steps >= agent.max_steps:
                break
        scores.append(np.max(score))
        scores_window.append(np.max(score))
        scores_avg.append(np.mean(scores_window))
        scores_std.append(np.std(scores_window))
        s_msg = "\rEpisode {}\tAverage Score: {:.3f}\tσ: {:.3f}\tScore: {:.3f}"
        print(s_msg.format(episode, np.mean(scores_window),
                           np.std(scores_window), np.max(score)), end="")
        if episode % 100 == 0:
            print(s_msg.format(episode, np.mean(scores_window),
                               np.std(scores_window), np.max(score)))
        if np.mean(scores_window) >= params["env_solved"]:
            solved = True
            s_msg = "\n\nEnvironment solved in {:d} episodes!\tAverage "
            s_msg += "Score: {:.3f}\tσ: {:.3f}"
            print(s_msg.format(episode, np.mean(scores_window),
                               np.std(scores_window)))
            # save the models
            s_name = agent.__name__
            s_aux = "%scheckpoint-%s.%s.%i.pth"
            for ii in range(len(agent)):
                s_actor_path = s_aux % (data_prefix, s_name, "actor", ii)
                s_critic_path = s_aux % (data_prefix, s_name, "critic", ii)
                torch.save(agent[ii].actor_local.state_dict(), s_actor_path)
                torch.save(agent[ii].critic_local.state_dict(), s_critic_path)
            break

    # save data to use later
    if not solved:
        s_msg = "\n\nEnvironment not solved =/"
        print(s_msg.format(episode, np.mean(scores_window),
                           np.std(scores_window)))
    print("\n")
    d_data = {"episodes": episode,
              "scores": scores,
              "scores_std": scores_std,
              "scores_avg": scores_avg,
              "scores_window": scores_window}
    s_aux = "%ssim-data-%s.data"
    pickle.dump(d_data, open(s_aux % (data_prefix, agent.__name__), "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Command line arguments ")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--path", required=True, help="path to environment app")
    parser.add_argument("--train", action="store_true", help="train the model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = load_yaml(f)

    env = UnityEnvironment(file_name=args.path)
    main(env, config, args.train)
