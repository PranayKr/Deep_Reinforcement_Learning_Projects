import numpy as np
import random
from collections import namedtuple ,deque
#from DDQN_NN_Model import DoubleDeepQNetModel
from NN_Model_Weights_Initialized import DeepQNetModel

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA= 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
#LR = 5e-4               # learning rate 
LR = 1e-4               # learning rate 
#LR = 5e-5               # learning rate 
#LR = 1e-5               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment.
    
    Attributes:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
    """
    
    def __init__(self,state_size, action_size,seed):
        #Initialize an Agent object.
        
        #Params
        #===========================================================================#
        #    state_size (int): dimension of each state
        #    action_size (int): dimension of each action
        #    seed (int): random seed
        #===========================================================================#
        
        self.state_size = state_size        
        self.action_size = action_size       
        self.seed = random.seed(seed)
        
        self.learning_rate = LR
        
         # Q-Network
        #self.Qnet_local = DoubleDeepQNetModel(state_size,action_size,seed).to(device)      
        #self.Qnet_target = DoubleDeepQNetModel(state_size,action_size,seed).to(device)    
            
        self.Qnet_local = DeepQNetModel(state_size,action_size,seed).to(device)      
        self.Qnet_target = DeepQNetModel(state_size,action_size,seed).to(device)
        #self.optimizer = optim.Adam(self.Qnet_local.parameters(),lr = LR)
        
        self.optimizer = optim.Adam(self.Qnet_local.parameters(),lr = self.learning_rate)
        # Replay memory
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)        
        self.t_step = 0
               
    def step(self,state,action,reward,next_state,done):      
        # saving experience in replay memory      
        self.memory.add(state,action,reward,next_state,done)        
        # learning every UPDATE_EVERY time steps         
        self.t_step = (self.t_step + 1)%UPDATE_EVERY
        if(self.t_step == 0):
            # If enough samples are available in memory, get random subset and learn
            if(len(self.memory) > BATCH_SIZE):
                experiences = self.memory.sample()
                self.learn(experiences,GAMMA)
            
    def act(self,state, eps = 0.) :
        
        # returns actions for current state as per given policy
        
        #Params
        #==========================================================#
        #   state (array_like): current state
        #   eps (float): epsilon, for epsilon-greedy action selection
        #==========================================================#
                                                                  
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)                       
        self.Qnet_local.eval()              
        with torch.no_grad():
            action_vals = self.Qnet_local(state)        
        self.Qnet_local.train()
        
        # Epsilon=greedy Action selection         
        if(random.random() > eps) :            
            return np.argmax(action_vals.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
                                     
                
    def learn(self,experiences,gamma):
        #updating value parameters given batch of experience tuples
        
        #Params
        #===========================================================================#
        #    experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
        #    gamma (float): discount factor
        #===========================================================================#
        
        states, actions, rewards, next_states , dones = experiences
        
        rewards_ = torch.clamp(rewards, min=-1., max=1.)
        
        #greedy actions (for next states) from local model       
        #qnet_local_greedy_action = self.Qnet_local(next_states).detach().argmax(dim=1).unsqueeze(1)
        # maximum predicted Q values for next states from target model indexed using greday action obtained from local model
        #Q_targets_next = self.Qnet_target(next_states).gather(1,qnet_local_greedy_action).detach()
        
        #greedy actions (for next states) from local model       
        qnet_local_greedy_action = self.Qnet_local(next_states).detach().max(1)[1].unsqueeze(1)
        # maximum predicted Q values for next states from target model indexed using greday action obtained from local model
        Q_targets_next = self.Qnet_target(next_states).gather(1,qnet_local_greedy_action)   
        
        
        # Q targets computation for current state         
        #Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))       
        
        
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones))       
        
        # Expected Q values from Local NN model        
        Q_expected = self.Qnet_local(states).gather(1,actions)       
        # loss computation       
        loss=  F.mse_loss(Q_expected,Q_targets)       
        #Minimize the loss       
        self.optimizer.zero_grad()
        loss.backward()       
        self.optimizer.step()       
        # updating target network        
        self.soft_update(self.Qnet_local,self.Qnet_target,TAU)
        
    def soft_update(self, local_model , target_model , tau ):
        
        # soft update model parameters
        #θ_target = τ*θ_local + (1 - τ)*θ_target
        
        #Params
        #==========================================================#
        #    local_model (PyTorch model): weights will be copied from
        #    target_model (PyTorch model): weights will be copied to
        #    tau (float): interpolation parameter 
        #==========================================================#
        
        for target_param , local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1 - tau) * target_param.data)
    
    
class ReplayBuffer:
    #Fixed-size buffer to store experience tuples.
    
    def __init__(self,action_size,buffer_size,batch_size, seed ):
        #Initialize a ReplayBuffer object.

        #Params
        #====================================================#
        #    action_size (int): dimension of each action
        #    buffer_size (int): maximum size of buffer
        #    batch_size (int): size of each training batch
        #    seed (int): random seed
        #====================================================#
        
        self.action_size = action_size        
        self.memory = deque(maxlen = buffer_size)       
        self.batch_size = batch_size        
        self.experience = namedtuple("Experience",field_names = ["state","action","reward","next_state","done"])        
        self.seed = random.seed(seed)
        
    def add(self,state,action, reward,mext_state,done):        
        # adding a new experience to memory for experience replay logic         
        exp = self.experience(state,action,reward,mext_state,done)        
        self.memory.append(exp)
        
    def sample(self):        
        # random sampling of a batch of experiences from memory         
        experiences = random.sample(self.memory ,k= self.batch_size)       
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        # cuurent size of memory        
        memmsize = len(self.memory)        
        return memmsize