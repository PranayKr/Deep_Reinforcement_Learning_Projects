
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
import numpy as np


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Tennis.app"`
# - **Windows** (x86): `"path/to/Tennis_Windows_x86/Tennis.exe"`
# - **Windows** (x86_64): `"path/to/Tennis_Windows_x86_64/Tennis.exe"`
# - **Linux** (x86): `"path/to/Tennis_Linux/Tennis.x86"`
# - **Linux** (x86_64): `"path/to/Tennis_Linux/Tennis.x86_64"`
# - **Linux** (x86, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86"`
# - **Linux** (x86_64, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Tennis.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Tennis.app")
# ```

# In[2]:


#env = UnityEnvironment(file_name="Tennis_Windows_x86_64\Tennis_Windows_x86_64\Tennis.exe")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
#brain_name = env.brain_names[0]
#brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# 
# The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
#env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
#num_agents = len(env_info.agents)
#print('Number of agents:', num_agents)

# size of each action
#action_size = brain.vector_action_space_size
#print('Size of each action:', action_size)

# examine the state space 
#states = env_info.vector_observations
#state_size = states.shape[1]
#print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
#print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agents and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agents' performance, if they select actions at random with each time step.  A window should pop up that allows you to observe the agents.
# 
# Of course, as part of the project, you'll have to change the code so that the agents are able to use their experiences to gradually choose better actions when interacting with the environment!

# In[5]:


#for i in range(1, 6):                                      # play game for 5 episodes
    #env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    #states = env_info.vector_observations                  # get the current state (for each agent)
    #scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    #while True:
        #actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        #actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        #env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        #next_states = env_info.vector_observations         # get next state (for each agent)
        #rewards = env_info.rewards                         # get reward (for each agent)
        #dones = env_info.local_done                        # see if episode finished
        #scores += env_info.rewards                         # update the score (for each agent)
        #states = next_states                               # roll over states to next time step
        #if np.any(dones):                                  # exit loop if episode finished
            #break
    #print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


# When finished, you can close the environment.

# In[6]:


#env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```

# In[7]:


from MADDPG_Agent import MADDPG
import torch
from collections import deque
from matplotlib import pyplot as plt


# In[8]:


def Train_MADDPGNetwork(env,brain_name,agent,num_agents,num_episodes=5000,max_timesteps=1000,print_every=100,train_mode=True):
    
    scores = []
    scores_deque = deque(maxlen=100)
    scores_avg = []
    
    
    for episode in range(1, num_episodes+1):
        scorestab = np.zeros(num_agents)                   # initialize the score (for each agent)
        rewardslist = []
        env_info = env.reset(train_mode=False)[brain_name]    # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)

        # loop over steps
        for t in range(max_timesteps):
            # select an action
            actions = agent.act(states,episode)
            # take action in environment and set parameters to new values
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scorestab += env_info.rewards                         # update the score (for each agent)
            # update and train agent with returned information
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            rewardslist.append(rewards)
            if np.any(dones):
                break

        # calculate episode reward as maximum of individually collected rewards of agents
        episode_reward = np.max(np.sum(np.array(rewardslist),axis=0))
        
        scores.append(episode_reward)             # save most recent score to overall score array
        scores_deque.append(episode_reward)       # save most recent score to running window of 100 last scores
        current_avg_score = np.mean(scores_deque)
        scores_avg.append(current_avg_score)      # save average of last 100 scores to average score array
    
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(episode, current_avg_score),end="")
        
        #print("\n")
    
        #print('Score (max over agents) from episode {}: {}'.format(episode, np.max(scorestab)))
    
        #print("\n")
    
        # log average score every 100 episodes
        if episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(episode, current_avg_score))
            
            print("\n")
    
            print('Score (max over agents) from episode {}: {}'.format(episode, np.max(scorestab)))
    
            print("\n")
       
        # break and report success if environment is solved
        if np.mean(scores_deque)>=.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(episode, np.mean(scores_deque)))
            agent.save_agents()
            break
    
    return scores


# In[9]:


def mainfunc():
    env = UnityEnvironment(file_name='Tennis_Windows_x86_64/Tennis_Windows_x86_64/Tennis.exe')
    print(type(env))
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  
    print(type(brain_name))
    print(brain_name)    
    print(brain)
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    
    MADDPG_Agent = MADDPG(seed=2, noise_start=0.5, update_every=2, gamma=0.99, t_stop_noise=30000)   
    
    print(type(MADDPG_Agent))
    
    training_scores = Train_MADDPGNetwork(env,brain_name,MADDPG_Agent,num_agents)
    
    
    #plotting the scores
    fig = plt.figure()
    ax= fig.add_subplot(111)
    plt.plot(np.arange(1,len(training_scores) + 1),training_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode Number')
    plt.show()
          
   


# In[10]:


if __name__ == "__main__":
    mainfunc()

