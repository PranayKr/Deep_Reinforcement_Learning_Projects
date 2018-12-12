# Implementation of Deep Deterministic Policy Gradient(DDPG) Algorithm to control a double-jointed Robotic Arm to consistently reach out to dynamically moving target location using Reacher Unity ML-Agent Environment , Multi-Layer Feedforward Neural Network with Pytorch Library 
# Problem Statement Description 
For this project, the task is to train a double-jointed Robotic Arm which can reach out to target locations (green sphere revolving around the robotic arm in the simulation environment) to maintain its position at the target location for as many timesteps as possible even as the target moves dynamically changing its position real-time. A reward of + 0.1 is provided for each timestep that the robotic arm agent is at the goal location.
# Results Showcase :
![reacher_single_untrained_agent](https://user-images.githubusercontent.com/25223180/49904809-68659180-fe91-11e8-8ecd-2cfbab77e3eb.gif)
# An Untrained Robotic Arm Agent taking random actions failing to reach out to the moving Target location 
![reacher_single_trained_agent](https://user-images.githubusercontent.com/25223180/49905013-33a60a00-fe92-11e8-8d05-c0be95ba0d75.gif)
# A Trained Robotic Arm Agent consistently reaching out to the moving Target location
# State Space :
The observation space consists of 33 variables corresponding to position,rotation,velocity and angular velocities of the double-jointed Robotic Arm 
# Action Space :
The Action Sapce is continuous. Each action ia a vector with 4 numbers (size:4) corresponding to torque applicable to two joints of the Robotic Arm. Every entry in action vector should be in the range of (-1,1)
# Solution Criteria :
# 1) Single-Agent Reacher Environment : 
     Only one robotic arm agent is present and the task is episodic.
     The environment is considered as solved when the agent gets an average score of +30 over 100 consecutive episodes. 
# 2) Multi-Agent Reacher Environment :
     There are 20 identical robotic arm agents each with its own copy of the environment
     The enviornment is considered as solved when all the 20 agents collectively get an average score of +30 over 100 consecutive episodes
