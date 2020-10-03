## Multiagent DDPG Using PyTorch for the ML-agent Tennis Environment

### Introduction
This repository is an implementation of the DDPG algorithm for Multi-agent Reinforcement Learning (MARL) the Tennis Environment developed by Unity3D and accessed through the UnityEnvironment library. It is an extension of the code sample provided by the Udacity Deep RL teaching crew (for more information visit their [website](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)). The environment is presented as a vector; thus, we did not use Convolutional Neural Networks (CNN) in the implementation.

This repository consists of these files:

*These files are saved under the "src" directory.*
1. <ins> model.py </ins>: This module provides the underlying neural network for our agent. When we train our agent, this neural network is going to be updated by backpropagation.
2. <ins>replay_buffer.py</ins>: This module implements the "memory" of our agent, also known as the Experience Replay.
3. <ins>agent.py</ins>: This is the body of our agent. It implements the way the agent acts using an actor-critic paradigm, and learn an optimal policy.
4. <ins>train.py</ins>: This module has the train function which takes the agent, the environment, number of training episodes and the required hyper-parameters and trains the agent accordingly. It also allows to test agents by passing `False` to its `train` parameter.

To test the code, after cloning the project, open the `Tennis.ipynb` notebook. It has all the necessary steps to install and load the packages, and train and test the agent. It also automatically detects the operating system, and loads the corresponding environment. There are two already trained agents stored as `{ag_1/ag_2}_checkpoint-actor.pth` and `{ag_1/ag_2}_checkpoint-critic.pth`, by running the last part of the notebook, this can be directly tested.

### The Tennis Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play

<img src="https://github.com/FredAmouzgar/Multiagent_DDPG_PyTorch_Tennis/raw/master/images/Tennis.jpeg" width="400" height="200">

### Multiagent Traning
The Tennis environment contains two unity agents. Each agent needs to collects observations from itself and its co-player. The task is essentially a cooperative task in that both agents maximize reward by hitting the ball back and forth for as long as possible.

### State and Action Space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Installation

### Python
Anaconda Python 3.6 is required: Download and installation instructions here: https://www.anaconda.com/download/

Create (and activate) a new conda (virtual) environment with Python 3.6.

Linux or Mac:

```bash
conda create --name yourenvnamehere python=3.6

source activate yourenvnamehere
```
Windows:
```bash
conda create --name yourenvnamehere python=3.6

activate yourenvnamehere
```
Download and save this GitHub repository.

To install required dependencies (torch, ML-Agents trainers, etc.), open the `Tennis.ipynb` and run the first cell.

Note: Due to its intricacy, you may have to install PyTorch separatetly.

### Unity Environment
For this example project, you will not need to install Unity - this is because you can use a version of the Reacher's unity environment that is already built (compiled) as a standalone application.

You don't need to download the environments seperately, although they are available underder the `_compressed_files` folder. The `Tennis.ipynb` notebook detects the right environment for your OS (except Windows (32 bit)).

## Training
1. Activate the conda environment you created above

2. Change the directory to the 'yourpath/thisgithubrepository' directory.

3. Run the first cells to initiate the training.

4. After training two pairs of `checkpoint_actor.pth` and `checkpoint_critic.pth` files will be saved for each agent with `ag_n_` prefix which consist of the trained model weights

5. See the performance plot after the training.

For more information about the DDPG training algorithm and the training hyperparameters see the included `Report.md` file.

## A Smart Agent
Here is a reward plot acquired by the agent while learning. It surpasses +33 after around 125 episodes.

<img src="" width="400" height="200">

Look at it go:

<img src="https://github.com/FredAmouzgar/Multiagent_DDPG_PyTorch_Tennis/raw/master/images/Tennis.gif" width="400" height="200">