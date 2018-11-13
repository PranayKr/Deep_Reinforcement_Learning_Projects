# A brief introduction to the Problem Statement
  Using a simplified version of the Banana Collector Unity ML-Agent environment, the objective of the project is to train an agent to     navigate and collect only yellow bananas in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a     reward (i.e. penalty) of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas   as possible while avoiding blue bananas. The agent’s observation space is 37 dimensional and the agent’s action space is 4 dimensional   (forward, backward, turn left, and turn right). The task is episodic, and in order to solve the environment, the agent must get an       average score of +13 over 100 consecutive episodes.
  
  
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
 
     
# Description of the Learning Algorithms used  

1) Deep Q-Learning Algorithm :
   In modern Q-learning, the function Q is estimated using a neural network that takes a state as input
   and outputs the predicted Q-values for each possible action. It is commonly denoted with Q(S, A, θ),
   where θ denotes the network’s weights. The actual policy used for control can subsequently be
   derived from Q by estimating the Q-values for each action give the current state and applying an
   epsilon-greedy policy. Deep Q-learning simply means using multilayer feedforward neural networks or even
   Convolutional Neural Networks (CNNs) to handle raw pixel input
   
2) Double Deep Q-Learning Algorithm :
   Deep Q-Learning is based upon Q-learning algorithm with a deep neural network as the function approximator. However, one issue that      Q-learning suffers from is the over estimation of the TD target in its update equation. The expected value is always greater than or    equal to the greedy action of the expected value. As a result, Q-learning ends up overestimating the q-values thereby degrading          learning efficiency. To address it, we use the double Q-learning algorithm where there are two separate q-tables. And at each time      step, we randomly decide which q-table to use and use the greedy action from one q-table to evaluate the q-value of the other q-table
   

