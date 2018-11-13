import numpy as np
import random
from collections import namedtuple ,deque
from DDQN_NN_Model import DoubleDeepQNetModel
#from NN_Model_Weights_Initialized import DeepQNetModel

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

ALPHA = 0.6    #Prioritization Exponent
INIT_BETA = 0.4    #Importance Sampling Exponent 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """An agent which Interacts with and learns(using Double Deep Q-Learning along with Prioritized Experience Replay)
       from the environment.
    
    Attributes:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
    """
    
    def __init__(self,state_size, action_size,seed,alpha=ALPHA,beta_start=INIT_BETA,max_timesteps=1000):
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
        self.Qnet_local = DoubleDeepQNetModel(state_size,action_size,seed).to(device)      
        self.Qnet_target = DoubleDeepQNetModel(state_size,action_size,seed).to(device)    
            
        #self.Qnet_local = DeepQNetModel(state_size,action_size,seed).to(device)      
        #self.Qnet_target = DeepQNetModel(state_size,action_size,seed).to(device)
        
        #self.optimizer = optim.Adam(self.Qnet_local.parameters(),lr = LR)
        
        self.optimizer = optim.Adam(self.Qnet_local.parameters(),lr = self.learning_rate)
        # Replay memory        
        self.memory = PrioritizedExpReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,seed,alpha)
        
        self.alpha = alpha
        self.start_beta = beta_start       
        self.max_timesteps = max_timesteps
                
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
        
    def get_sampling_exp_val(self,timestep):
        
        exp_increment_factor = min((float(timestep)/self.max_timesteps), 1.0)
        
        new_beta = self.start_beta + (exp_increment_factor * (1- self.start_beta))
        
        return new_beta
                                     
                
    def learn(self,experiences,gamma,timestep=1000):
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
        
        
        # Importance Sampling Weights Computation
        
        current_Beta = self.get_sampling_exp_val(timestep)
        
        imp_sampling_wghts = self.memory.get_impsampling_weights(current_Beta)
        
        tempdiff_error = Q_targets - Q_expected
        
        self.memory.update_priorities(tempdiff_error)        
        
        # loss computation   
        # calculation of the weighted mse loss to be used by Prioritized experience replay   

        squareddiff_error = (Q_expected - Q_targets)**2
        squareddiff_error = squareddiff_error * imp_sampling_wghts.expand_as(squareddiff_error)
        weighted_loss = squareddiff_error.mean(0)  
    
        loss = weighted_loss
                               
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
    
    
class PrioritizedExpReplayBuffer:
    #Fixed-size buffer to store experience tuples.
    
    def __init__(self,action_size,buffer_size,batch_size, seed ,alpha):
        #Initialize a ReplayBuffer object.

        #Params
        #====================================================#
        #    action_size (int): dimension of each action
        #    buffer_size (int): maximum size of buffer
        #    batch_size (int): size of each training batch
        #    seed (int): random seed
        #    alpha (float) : Prioritization Exponent value ranging between 0-1
        #====================================================#
        
        self.action_size = action_size        
        self.memory = deque(maxlen = buffer_size)       
        self.batch_size = batch_size        
        self.experience = namedtuple("Experience",field_names = ["state","action","reward","next_state","done"])        
        self.seed = random.seed(seed)
        
        self.alpha = max(0.,alpha)
        
        self.priorities = deque(maxlen=buffer_size)
        
        self.buffer_size = buffer_size
        
        self.sumtotal_priorities = 0
        
        self.epsilon  = 1e-6
        
        self.indexes = []
        
        self.maxval_priority = 1.0**self.alpha
                
        
    def add(self,state,action, reward,mext_state,done):        
        # adding a new experience to memory for experience replay logic         
        exp = self.experience(state,action,reward,mext_state,done)        
        self.memory.append(exp)
        
        if(len(self.priorities) >= self.buffer_size):
            self.sumtotal_priorities -= self.priorities[0]
            
            
        self.priorities.append(self.maxval_priority)
        
        self.sumtotal_priorities += self.priorities[-1]
            
        
    def sample(self):        
        #sampling of a batch of experiences from memory according to importance sampling weights 
        
        replay_mem_len = len(self.memory)
        
        sampling_probs = None
        
        if(self.sumtotal_priorities >0):
            sampling_probs = np.array(self.priorities)/self.sumtotal_priorities
            
        exp_indices = np.random.choice(replay_mem_len , size = min(replay_mem_len,self.batch_size) , p =sampling_probs)
        
        self.indexes = exp_indices
        
        experiences = [self.memory[i] for i in self.indexes]
                    
        #experiences = random.sample(self.memory ,k= self.batch_size) 
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)        
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    
    def calculate_sample_weights(self,priority,beta,maxweight,mem_len):
        
        imp_smpling_weight = ((mem_len*(priority/self.sumtotal_priorities))**-beta)/maxweight
        
        return imp_smpling_weight
    
    
    def get_impsampling_weights(self,beta):
        
        #return importance sampling weights of experience sample using the value of Sampling exponent Beta
        
        replay_mem_len = len(self.memory)
        
        max_smpling_weight = (replay_mem_len * (min(self.priorities)/self.sumtotal_priorities))**-beta
        
        list_smpling_weights  = [self.calculate_sample_weights(self.priorities[i],beta,max_smpling_weight,replay_mem_len) for i in                                          self.indexes] 
        
        return torch.tensor(list_smpling_weights,device = device,dtype =torch.float).reshape(-1,1)
    
    
    def update_priorities(self,tempdiff_error):
        
        # updating priorities of samples based on temp diff errors of last samples
        
        for priority_index, td_error in zip(self.indexes,tempdiff_error):
            td_error = float(td_error)
            self.sumtotal_priorities -= self.priorities[priority_index]
            self.priorities[priority_index] = (abs(td_error) + self.epsilon)**self.alpha            
            self.sumtotal_priorities += self.priorities[priority_index]
        
        self.maxval_priority = max(self.priorities)
                
        self.indexes = []
            
    
    def __len__(self):
        # cuurent size of memory        
        memmsize = len(self.memory)        
        return memmsize