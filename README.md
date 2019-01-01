# Implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm to train 2 Agents play a game of Tennis against each other using modified version of Tennis Unity ML-Agent Environment and Multi-Layer Feedforward Neural Network Model with Pytorch Library

# Problem Statement Description
For this project the task is train 2 Agents play a game of Tennis against each other. In this environment, both agents control rackets to bounce a ball over a net. A reward of +0.1 is provided if an agent hits the ball over the net whereas a reward (penalty) of -0.01 is provided if an agent lets a ball hit the ground or hits the ball out of bounds. Thus, the goal of each agent is to keep the ball in play.

# Results Showcase :

![tennis_game_multi-agent_untrained](https://user-images.githubusercontent.com/25223180/50571868-5153f880-0dda-11e9-894b-5dee11e09b41.gif)

# Untrained Agents moving Rackets at random unable to hit the Tennis Ball

![tennis_game_multi-agent_trained](https://user-images.githubusercontent.com/25223180/50571890-b3146280-0dda-11e9-8f77-3ffcea5df3f2.gif)

# Trained Agents properly playing Tennis controlling Rackets to consistently hit the Tennis Ball over the Net

# State Space :
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.

# Action Space :
The Action Sapce is continuous. 2 continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

# Solution Criteria :
The task is episodic, and in order to solve the environment, both the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
1) After each episode, sum of the rewards that each agent received (without discounting) is taken to get a score for each agent. This yields 2 (potentially different) scores. 
2) Only the maximum of these 2 scores is considered for each episode.

The environment is considered solved, when the average of the maximum score per episode (over 100 episodes) is at least +0.5.


