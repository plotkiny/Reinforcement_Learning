


## Table of contents
1. [About](#about)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Details](#project)
5. [Future Work](#future)


## About <a name="about"></a>
>Trained an agent to navigate (and collect bananas!) in a large, square world. A reward of +1 is 
provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue 
banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while 
avoiding blue bananas.
>The state space has 37 dimensions and contains the agent's velocity, along with ray-based 
perception of objects around agent's forward direction. Given this information, the agent has to 
learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

>The task is episodic, and in order to solve the environment, your agent must get an average score 
of +13 over 100 consecutive episodes. Implemented DQN and DuelingDQN with priority experience 
replay, and both adaptive learning rate and beta hyperparameters. Model architecture consists of 
alternating layers and dropout for preventing overfitting. Number of hidden nodes can be 
specificed as a hyperparameter.


## Installation <a name="installation"></a>
>General Requirements: Create the conda environment used for the training of the DQN using the requirements.txt file.
>Use the requirements.txt file for installing dependencies

## Usage <a name="usage"></a>
>The command-line arguments used for training include both the --config (supplies
model hyperparameters for training) and --path arguments (path to the Banana.app). Both are REQUIRED. Optional 
boolean --train parameter for training the model.

>Train the DQN model commad
```bash
python run.py --path path_to_repository/deep-reinforcement-learning/p1_navigation/Banana.app --config config.txt --train
```

## Project Details <a name="project"></a>
* config.txt file used for specifying hyperparameters
* navigation_agent.py file for defining the agent class
* navigation_model.py file for defining the DQN networks
* run.py file for training the agent using the command line




