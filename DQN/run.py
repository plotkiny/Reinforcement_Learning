#!usr/bin/env python

############################################################
# Copyright (C) 2019 Yuri Plotkin (plotkiny@gmail.com)     #
# Permission given to modify the code as long as you keep  #
# this decleration at the top                              #
############################################################


import argparse
import numpy as np
import torch
import yaml
import yamlordereddictloader
from collections import deque
from navigation_agent import Agent
from unityagents import UnityEnvironment


def get_environment_info(env, brain, brain_name):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)

    state_size = len(state)
    print('States have length:', state_size)

    return state_size, action_size


def load_yaml(file_obj):
    return yaml.load(file_obj, Loader=yamlordereddictloader.Loader)


def main(config, train, env, brain_name, state_size, action_size):

    params = config["parameters"]
    eps_start = params["eps_start"]
    eps_end = params["eps_end"]
    eps_decay = params["eps_decay"]
    hidden_layers = params["hidden_layers"]
    n_episodes = params["n_episodes"]
    seed = params["seed"]

    agent = Agent(params, state_size, action_size, seed, hidden_layers)

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = agent.act(state, i_episode, eps) # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done, i_episode)
            state = next_state # roll over the state to next time step
            score += reward  # update the score
            if done: # exit loop if episode finished
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)))
        if i_episode % 25 == 0:
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 20.0:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break

    return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Command line arguments ")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--path", required=True, help="path to environment app")
    parser.add_argument("--train", action="store_true", help="train the model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = load_yaml(f)

    env = UnityEnvironment(file_name=args.path)

    #get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    #get environment info
    state_size, action_size = get_environment_info(env, brain, brain_name)

    #run network
    scores = main(config, args.train, env, brain_name, state_size, action_size)
