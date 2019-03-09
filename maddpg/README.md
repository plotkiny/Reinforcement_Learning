


## Table of contents
1. [About](#about)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Details](#project)
5. [Future Work](#future)


## About <a name="about"></a>
>In this environment, two agents control rackets to bounce a ball over a net. If an agent
hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the 
ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each 
agent is to keep the ball in play.
>The observation space consists of 8 variables corresponding to the position and velocity 
of the ball and racket. Each agent receives its own, local observation. Two continuous 
actions are available, corresponding to movement toward (or away from) the net, and jumping.


>The task is episodic, and in order to solve the environment, your agents must get an 
average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both 
agents). Specifically,

    * After each episode, we add up the rewards that each agent received (without discounting), 
    to get a score for each agent. This yields 2 (potentially different) scores. We then 
    take the maximum of these 2 scores.
    
    * This yields a single score for each episode.

>The environment is considered solved, when the average (over 100 episodes) of those scores 
is at least +0.5.


## Installation <a name="installation"></a>
>General Requirements: Create the conda environment used for the training of the Actor-Critic using the 
requirements.txt file.
>Use the requirements.txt file for installing dependencies

## Usage <a name="usage"></a>
>The command-line arguments used for training include both the --config (supplies
model hyperparameters for training) and --path arguments (path to the Banana.app). Both are REQUIRED. Optional 
boolean --train parameter for training the model.

>Train the Multi-Agent DDPG model commad
```bash
python run.py --path path_to_repository/deep-reinforcement-learning/Tennis.app --config config.txt --train
```

## Project Details <a name="project"></a>
* config.txt file used for specifying hyperparameters
* maddpg.py file for defining the agent class
* ddpg.py file for defining the Multi-Agent DDPG networks
* run.py file for training the agent using the command line




