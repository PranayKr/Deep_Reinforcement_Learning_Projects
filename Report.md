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
     
  b) Q-Learning:
     Q-learning is an off-policy Temporal Difference (TD) Control algorithm. Off-policy methods evaluate
     or improve a policy that differs from the policy used to make decisions. These decisions can thus be
     made by a human expert or random policy, generating (state, action, reward, new state) entries to
     learn an optimal policy from.
     
  c) Temporal Difference Learning :
     Temporal Difference learning is a central idea to modern day RL and works by updating estimates for the 
     action-value function based on other estimates. This ensures the agent does not have to wait until the 
     actual cumulative reward after completing an episode to update its estimates, but is able to learn from 
     each action.
     


# Description of the Learning Algorithms used  

1) Deep Q-Learning Algorithm 
   

