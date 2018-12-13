# A brief introduction to the Problem Statement
  Using a Single-Agent version of the Multi-Agent Reacher Unity ML-Agent environment, the objective of the project is to train a double     jointed Robotic Arm Agent which can reach out to target locations (green sphere revolving around the robotic arm) to maintain its         position at the target location for as many timesteps as possible even as the target moves dynamically changing its position real-time.
  A reward of + 0.1 is provided for each timestep that the robotic arm agent is at the goal location. The agent’s observation space         consists of 33 variables corresponding to position,rotation,velocity and angular velocities of the double-jointed Robotic Arm.
  The agent's action space is continuous. Each action ia a vector with 4 numbers (size:4) corresponding to torque applicable to two
  joints of the Robotic Arm. Every entry in action vector should be in the range of (-1,1). The task is episodic, and in order to solve     the environment, the agent must get an average score of +30 over 100 consecutive episodes.
  
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
    In the Q-learning algorithm using a function approximator, the TD target is also dependent on the network parameter w that is           being learnt/updatet, and this can lead to instabilities. To address it, a separate network with identical architecture but             different weights is used. And the weights of this separate target network are updated every few steps to be equal to the local         network that is continuously being updated.
    
 f) Deep Q-Learning Algorithm :
    In modern Q-learning, the function Q is estimated using a neural network that takes a state as input
    and outputs the predicted Q-values for each possible action. It is commonly denoted with Q(S, A, θ),
    where θ denotes the network’s weights. The actual policy used for control can subsequently be
    derived from Q by estimating the Q-values for each action give the current state and applying an
    epsilon-greedy policy. Deep Q-learning simply means using multilayer feedforward neural networks or even
    Convolutional Neural Networks (CNNs) to handle raw pixel input
    
 g) Value-Based Methods :
    Value-Based Methods such as Q-Learning and Deep Q-Learning aim at learning optimal policy from interaction with the environment
    by trying to find an estimate of the optimal action-value function . While Q-Learning is implemented for environments having small       state spaces by representing optimal action-value function in the form of Q-table with one row for each state and one column for         each action which is used to build the optimal policy one state at a time by pulling action with maximum value from the row             corresponding to each state;it is impossible to maintain a Q-Table for environments with huge state spaces in which case the optimal
    action value function is represented using a non-linear function approximator such as a neural network model which forms the basis       of Deep Q-Learning algorithm. 
    
 h) Policy-Based Methods :
    Unlike in the case of Value-Based Methods the optimal policy can be found out directly from interaction with the environment without
    the need of first finding out an estimate of optimal action-value function by using Policy-Based Methods. For this a neural network
    is constructed for approximating the optimal policy which takes in all the states in state space as input(number of input neurons       being equal to number of states) and returns the probability of each action being selected in action space (number of output neurons     being equal to number of actions in action space) The agent uses this policy to interact with the environment by passing only the       most recent state as input to the neural-net model.Then the agent samples from the action probabilities to output an action in           response.The algorithm needs to optimize the network weights ao that the optimal action is most likely to be selected for each 
    iteraion during training, This logic helps the agent with its goal to maximize expected return.
    

# Description of the Learning Algorithm used  
