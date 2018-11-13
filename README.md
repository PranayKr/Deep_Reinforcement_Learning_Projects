# Implementation of Vanilla Deep Q-Learning / Double Deep Q-Learning and Double Deep Q-Learning with Prioritized Experience Replay  Algorithms to solve Banana-Collector Unity ML-Agent Navigation Problem Statement using Multi-Layer Feedforward Neural Network Model with Pytorch Library

# Problem Statement Description 
For this project, the task is to train an agent to navigate in a large, square world, while collecting yellow bananas, and avoiding blue bananas. A reward of +1 is provided for collecting a yellow banana, and a reward(i.e. penalty) of -1 is provided for collecting a blue banana. Thus, the goal is to collect as many yellow bananas as possible while avoiding blue bananas.

# Results Showcase :
![banana-collector_untrained_agent](https://user-images.githubusercontent.com/25223180/48391940-00efe100-e72f-11e8-9710-7e425a0e6dbd.gif)
# An Untrained Agent navigating at random in the environment

![banana-collector_trained_agent](https://user-images.githubusercontent.com/25223180/48392534-cc315900-e731-11e8-95e1-27253fbc07cd.gif)
# A Trained Agent navigating with the goal of collecting yellow bananas and avoiding blue bananas

# State Space : 
The observations are in a 37-dimensional continuous space corresponding to 35 dimensions of ray-based perception of objects around the agent’s forward direction and 2 dimensions of velocity. The 35 dimensions of ray perception are broken down as: 7 rays projecting from the agent at the following angles (and returned back in the same order): [20, 90, 160, 45, 135, 70, 110] where 90 is directly in front of the agent. Each ray is 5 dimensional and it is projected onto the scene. If it encounters one of four detectable objects (i.e. yellow banana, wall, blue banana, agent), the value at that position in the array is set to 1. Finally there is a distance measure which is a fraction of the ray length. Each ray is [Yellow Banana, Wall, Blue Banana, Agent, Distance].
The velocity of the agent is two dimensional: left/right velocity and forward/backward velocity.
The observation space is fully observable because it includes all the necessary information regarding the type of obstacle, the distance to obstacle, and the agent’s velocity. As a result, the observations need not be augmented to make them fully observable.The incoming observations can thus be directly used as state representation.

# Action Space :
The action space is 4 dimentional. Four discrete actions correspond to:
a) 0 - move forward
b) 1 - move backward
c) 2 - move left
d) 3 - move right

# Solution Criteria :
The environment is considered as solved when the agent gets an average score of +13 over 100 consecutive episodes.

# Installation Instructions to setup the Unity ML environment :
# 1) Setting Up Python Environment :
     a) Create (and activate) a new environment with Python 3.6.:
     
        Linux or Mac:
        conda create --name drlnd python=3.6
        source activate drlnd
        
        Windows:
        conda create --name drlnd python=3.6 
        activate drlnd
        
     b) Minimal Installation of OpenAi Gym Environment
        Below are the instructions to do minimal install of gym :

        git clone https://github.com/openai/gym.git
        cd gym
        pip install -e .
         
        A minimal install of the packaged version can be done directly from PyPI:

        pip install gym
         
     c) Clone the repository and navigate to the python/ folder. Then, install several dependencies.
          
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install . (or pip install [all] )
          
     d) Create an Ipython Kernel for the drlnd environment :
          
        python -m ipykernel install --user --name drlnd --display-name "drlnd"
          
     e) Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.
     
     
          

