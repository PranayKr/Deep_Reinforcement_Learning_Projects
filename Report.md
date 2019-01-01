# A brief introduction to the Problem Statement
Using modified version of Tennis Unity ML-Agent Environment, the objective of the project is to train 2 Agents play a game of Tennis against each other. In this environment, both agents control rackets to bounce a ball over a net. A reward of +0.1 is provided if an agent hits the ball over the net whereas a reward (penalty) of -0.01 is provided if an agent lets a ball hit the ground or hits the ball out of bounds. Thus, the goal of each agent is to keep the ball in play. The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. The Action Sapce is continuous. 2 continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.The task is episodic, and in order to solve the environment, both the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

1) After each episode, sum of the rewards that each agent received (without discounting) is taken to get a score for each agent. This yields 2 (potentially different) scores.
2) Only the maximum of these 2 scores is considered for each episode.

The environment is considered solved, when the average of the maximum score per episode (over 100 episodes) is at least +0.5.

# Relevant Concepts
This section provides a theoretical background describing current work in this area as well as concepts
and techniques used in this work.
  
a) Reinforcement Learning:
   Reinforcement Learning has become very popular with recent breakthroughs such as AlphaGo
   and the mastery of Atari 2600 games. Reinforcement Learning (RL) is a framework for learning
   a policy that maximizes an agent’s long-term reward by interacting with the environment. A policy
   maps situations (states) to actions. The agent receives an immediate short-term reward after each state
   transition. The long-term reward of a state is specified by a value-function. The value of state roughly
   corresponds to the total reward an agent can accumulate starting from that state. The action-value
   function corresponding to the long-term reward after taking action a in state s is commonly referred
   to as Q-value and forms the basis for the most widely used RL technique called Q-Learning
 
b) Temporal Difference Learning :
   Temporal Difference learning is a central idea to modern day RL and works by updating estimates for the 
   action-value function based on other estimates. This ensures the agent does not have to wait until the 
   actual cumulative reward after completing an episode to update its estimates, but is able to learn from 
   each action.
      
c) Q-Learning:
   Q-learning is an off-policy Temporal Difference (TD) Control algorithm. Off-policy methods evaluate
   or improve a policy that differs from the policy used to make decisions. These decisions can thus be
   made by a human expert or random policy, generating (state, action, reward, new state) entries to
   learn an optimal policy from.
     
   Q-learning learns a function Q that approximates the optimal action-value function. It does this by randomly 
   initializing Q and then generating actions using a policy derived from Q, such as e-greedy. An e-greedy policy 
   chooses the action with the highest Q value or a random action with a (low) probability of , promoting exploration 
   as e (epsilon) increases. With this newly generated (state (St), action (At), reward (Rt+1), new state (St+1)) pair,
   Q is updated using rule 1.
     
   This update rule essentially states that the current estimate must be updated using the received immediate reward 
   plus a discounted estimation of the maximum action-value for the new state. It is important to note here that the 
   update is done immediately after performing the action using an estimate instead of waiting for the true cumulative
   reward, demonstrating TD in action. The learning rate α decides how much to alter the current estimate and the
   discount rate γ decides how important future rewards (estimated action-value) are compared to the immediate reward.
     
d) Experience Replay:
   Experience Replay is a mechanism to store previous experience (St, At, Rt+1, St+1) in a fixed size buffer. 
   Minibatches are then randomly sampled, added to the current time step’s experience and used to incrementally 
   train the neural network. This method counters catastrophic forgetting, makes more efficient use of data by 
   training on it multiple times, and exhibits better convergence behaviour
    
e) Fixed Q Targets :
   In the Q-learning algorithm using a function approximator, the TD target is also dependent on the network parameter w that is            being learnt/updatet, and this can lead to instabilities. To address it, a separate network with identical architecture but              different weights is used. And the weights of this separate target network are updated every few steps to be equal to the local          network that is continuously being updated.
    
f) Deep Q-Learning Algorithm :
   In modern Q-learning, the function Q is estimated using a neural network that takes a state as input
   and outputs the predicted Q-values for each possible action. It is commonly denoted with Q(S, A, θ),
   where θ denotes the network’s weights. The actual policy used for control can subsequently be
   derived from Q by estimating the Q-values for each action give the current state and applying an
   epsilon-greedy policy. Deep Q-learning simply means using multilayer feedforward neural networks or even
   Convolutional Neural Networks (CNNs) to handle raw pixel input
    
g) Value-Based Methods :
   Value-Based Methods such as Q-Learning and Deep Q-Learning aim at learning optimal policy from interaction with the environment
   by trying to find an estimate of the optimal action-value function . While Q-Learning is implemented for environments having small      state spaces by representing optimal action-value function in the form of Q-table with one row for each state and one column for        each action which is used to build the optimal policy one state at a time by pulling action with maximum value from the row              corresponding to each state ; it is impossible to maintain a Q-Table for environments with huge state spaces in which case the          optimal action value function is represented using a non-linear function approximator such as a neural network model which forms
   the basis of Deep Q-Learning algorithm. 
    
h) Policy-Based Methods (for Discrete Action Spaces) :   
   Unlike in the case of Value-Based Methods the optimal policy can be found out directly from interaction with the environment without
   the need of first finding out an estimate of optimal action-value function by using Policy-Based Methods. For this a neural network
   is constructed for approximating the optimal policy which takes in all the states in state space as input(number of input neurons        being equal to number of states) and returns the probability of each action being selected in action space (number of output neurons    being equal to number of actions in action space) The agent uses this policy to interact with the environment by passing only the        most recent state as input to the neural-net model.Then the agent samples from the action probabilities to output an action in          response.The algorithm needs to optimize the network weights ao that the optimal action is most likely to be selected for each 
   iteraion during training, This logic helps the agent with its goal to maximize expected return.
    
   NOTE : The above mentioned process explains the way to appoximate a Stochastic Policy using Neural-Net by sampling of action
          probabilities. Policy-Based Methods can be used to approximate a Deterministic Policy as well by selecting only the
          Greedy Action for the latest input state fed to the model during forward pass for each iteration during training
 
i) Policy-Based Methods (for Continuous Action Spaces) :
   Policy-Based Methods can be used for environments having continuous action space by using a neural network used for
   estimating the optimal policy having an output layer which parametrizes a continuous probability distribution by 
   outputting the mean and variance of a normal distribution. Then in order to select an action , the agent needs to 
   pass only the most recent state as input to the network and then use the output mean and variance to sample from the 
   distribution.
     
   Policy-Based Methoda are better than the Value-Based Methods owing to the following factors:
   a) Policy-Based Methods are simpler than Value-Based Methods because they do away with the intermediate step of estimating
      the optimal action-value function and directly estimate the optimal policy.
   b) Policy-Based Methods tend to learn the true desired stochastic policy whereas Value-Based Methods tend to learn a
      deterministic or near-deterministic policy.
   c) Policy-Based Methods are best suited for estimating optimal policy for environments with Continuous Action-Spaces as they
      directly map a state to action unlike Value-Based Methods which need to first estimate the best action for each state which
      can be carried out provided that the action space is discrete with finite number of actions ; but in case of Continuous 
      Action Space the Value-Based Methods need to find the global maximum of a non-trivial continuous action function which 
      turns out to be an Optimization Problem in itself.
    
j) Policy-Gradient Methods :
   Policy-Gradient Methods are a subclass of Policy-Based Methods which estimate the weights of an optimal policy by first
   estimating the gradient of the Expected Return (cumulative reward) over all trajectories as a function of network weights
   of a Neural-Net Model representing policy PI to be optimized using Stochastic Gradient Ascent by looking at each
   state-action pair in a Trajectory separately and by taking into account the magnitude of cumulative reward i.e. expected
   return for that Trajectory. Either network weights are updated to increase the probability of selecting an action for a
   particular state in case of receiving a positive reward or the network weights are updated to decrease the probability of
   selecting an action for a partcular state in case of receiving a negative reward by calculating the gradient i.e. derivative
   of the log of probability of selecting an action given a state using the Policy PI with weights Theta and multiplying it with
   the reward (expected return) received over all state-action pairs present in a trajectory summed up over all the sampled set 
   of trajectories.The weights Theta of the policy are now updated with the gradient estimate calculated over several iterations
   with the final goal of converging to the weights of an optimal policy.
 
k) Actor-Critic Methods :
   Actor-Critic Methods are at the intersection of Value-Based Methods such as Deep-Q Network and Policy-Based Methods such as 
   Reinforce. They use Value-Based Methods to estimate optimal action-value function which is then used as a baseline to reduce
   the variance of Policy-Based Methods. Actor-Critic Agents are more stable than Value-Based Agents and need fewer samples/data
   to learn than Policy-Based Agents. Two Neural Networks are used here one for the Actor and one for the Critic. The Critic 
   Neural-Net takes in a state to output State-Value function of Policy PI. The Actor Neural Net takes in a state and outputs
   action with highest probability to be taken for that state which is used then to calculate TD-Estimate for current state by
   using the reward for current state and the next state. The Critic Neural-Net gets trained using this TD-Estimate value. The
   Critic Neural-Net then is used to calculate the Advantage Function (sum of reward for current state and difference of discounted
   TD-Estimate of next state and TD-Estimate of current state) which is then used as a baseline to train the Actor Neural-net. 
 
l) Deep Deterministic Policy Gradient (DDPG) Algorithm :
   DDPG Algorithm can be best described as Deep-Q Network Method for continuous action spaces instead of being called an actual
   Actor-Critic Method. The Critic Neural-Net Model in DDPG Algorithm is used to approximate the maximiser over the Q-Values of
   the next state and not as a learned baseline to reduce variance of the Actor Neural-Net Model. A Deep Q-Network Method cannot
   be used for continuous action spaces. In DDPG Method 2 neural networks are used for actor and critic similar to a basic
   Actor-Critic Method. The Actor Neural-Net is used to approximate the optimal policy deterministically by outputting only the
   best action for a given state rather than probability distribution for each action. The Critic Neural-Net estimates the optimal
   action-value function for a given state using the best-believed action outputted by the Actor Neural-Net. Hence here Actor is 
   being used as an approximate maximiser to calculate a new target value for training the Critic to estimate action-value function
   for that state. DDPG algorithm uses Replay Buffer to store and get sampled set of experience tuples and uses the concept of
   soft-update to update the target networks of actor and critic as explained below.

   Soft-Update of Actor and Critic Network Weights :
   In DDPG Algorithm regular and target network weights are there for both actor and critic networks. Target Network Weights are 
   updated every time-step by adding a minor percentage of regular network weights to current target network weights keeping the
   major part of current target network weights. Using this soft-update strategy for updating target network weights helps a lot
   in accelerating learning.
    
m) Multi-Agent Reinforcement Learning (MARL) :
   When Multi-Agent Systems use Reinforcement Learning techniques to train the agents and make them learn their behaviours , then this
   process is termed as Multi-Agent Reinforcement Learning. Markov Games are a framework for implementation of MARL in the same way as
   Markov Decision Processes (MDPs) are used for Single-Agent Reinforcement Learning techniques.
    
n) Markov Games :
   Markov Games are a framework for implementation of Multi-Agent Reinforcement Learning (MARL) techniques in the same way as Markov 
   Decision Processes (MDPs) are used for Single-Agent Reinforcement Learning techniques.
   A Markov Game is a set of following parameters :
   a) Number of Agents 
   b) Set of Environment States
   c) Set of Actions of each Agent
   d) Combined Action Space of both the Agents
   e) Set of Observations of each Agent
   f) Reward function of each Agent
   g) Set of Policies of each Agent
   h) State Transition function provided the current State and joint Action of all the Agents
    
# Description of the Learning Algorithm used
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Algorithm : MADDPG Algorithm is an extension of the concept of DDPG Algorithm
for multiple Agents. Each Agent individually is trained using DDPG Algorithm with each agent having its own actor and critic model. 
DDPG Algorithm is an off-policy actor-critic algorithm that uses the concept of target networks which get updated using the latest
weights of local networks for both actor and critic. While implementing MADDPG, actor model of each Agent receive as input the individual state (observations) of the agent and output a (two-dimensional) action vector. The critic model of each agent however, receives the states and actions of all actors of all the agents. This approach leads to information sharing between the agents. During training the Critic of each Agent receives extra information such as states observed and actions taken by all other Agents present whereas the Actor of each Agent has information regarding only that pertcular Agent's observed states and actions taken. MADDPG Algorithm can be used to train multiple Agents in cooperative , competitive or mixed cooperative competitive environments. To summarize it can be said that MADDPG Algorithm is a centralized training and decentralized execution algorithm.

# Neural Net Architecture Used:
# a) Architecture of Actor Neural-Network Model (for each Agent) :
    A multilayer feed-forward Neural Net Architecture was used with 2 Hidden layers each having 256 hidden neurons.The input layer
    has number of input neurons equal to the state size and the the output layer has number of output neurons equal to action size. A
    Leaky ReLU (Rectified Linear Unit) Activation Function was used over the inputs of the 2 hidden layers while a tanh activation 
    function was used over the input of the output laye. Batch Normalizatio was used over the output of the first hidden layer. Weight  
    initialization was done for the first 2  layers  from uniform distribution in the negative to positive range of reciprocol of the  
    square root of number of weights for each layer. Weight initialization for the final layer was done from uniform distribution in the 
    range of (-3e-3, 3e-3).

# b) Architecture of Critic Neural-Network Model (for each Agent) :

# c) Other Details of Implementation :

# HyperParameters Used:

# Plot of Rewards per Episode

# CONCLUSION

# Ideas for Future Works (Scope for further improvements)

# REFERENCES :



   
