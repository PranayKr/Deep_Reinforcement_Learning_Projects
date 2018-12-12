# Implementation of Deep Deterministic Policy Gradient(DDPG) Algorithm to control a double-jointed Robotic Arm to consistently reach out to dynamically moving target location using Reacher Unity ML-Agent Environment , Multi-Layer Feedforward Neural Network with Pytorch Library 
# Problem Statement Description 
For this project, the task is to train a double-jointed Robotic Arm which can reach out to target locations (green sphere revolving around the robotic arm in the simulation environment) to maintain its position at the target location for as many timesteps as possible even as the target moves dynamically changing its position real-time. A reward of + 0.1 is provided for each timestep that the robotic arm agent is at the goal location.
# Results Showcase :
![reacher_single_untrained_agent](https://user-images.githubusercontent.com/25223180/49904809-68659180-fe91-11e8-8ecd-2cfbab77e3eb.gif)
# An Untrained Robotic Arm Agent taking random actions failing to reach out to the moving Target location 
![reacher_single_trained_agent](https://user-images.githubusercontent.com/25223180/49905013-33a60a00-fe92-11e8-8d05-c0be95ba0d75.gif)
# A Trained Robotic Arm Agent consistently reaching out to the moving Target location
