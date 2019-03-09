


## Table of contents
1. [About](#about)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Details](#project)
5. [Future Work](#future)


## About <a name="about"></a>
>Trained an agent to is to maintain its position at the target location for as many time steps as possible.
>The observation space consists of 33 variables corresponding to position, rotation, 
velocity, and angular velocities of the arm. Each action is a vector with four numbers, 
corresponding to torque applicable to two joints. Every entry in the action vector should be 
a number between -1 and 1.


>The task is episodic, and in order to solve the environment, your agent must get an average score 
of +30 over 100 consecutive episodes. Implemented Actor-Critic Network with priority experience 
replay, and both adaptive learning rate and hyperparameters for tuning model. Model architecture 
consists of alternating layers and dropout for preventing overfitting. Number of hidden nodes can be 
specificed as a hyperparameter.


## Installation <a name="installation"></a>
>General Requirements: Create the conda environment used for the training of the Actor-Critic using the requirements.txt file.
>Use the requirements.txt file for installing dependencies

## Usage <a name="usage"></a>
>The command-line arguments used for training include both the --config (supplies
model hyperparameters for training) and --path arguments (path to the Banana.app). Both are REQUIRED. Optional 
boolean --train parameter for training the model.

>Train the DDPG model commad
```bash
python run.py --path path_to_repository/deep-reinforcement-learning/Reacher.app --config config.txt --train
```

## Project Details <a name="project"></a>
* config.txt file used for specifying hyperparameters
* agent.py file for defining the agent class
* ddpg.py file for defining the DQN networks
* run.py file for training the agent using the command line




