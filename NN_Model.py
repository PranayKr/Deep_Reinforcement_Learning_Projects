import torch
import torch.nn as nn
import torch.nn.functional as F


class  DeepQNetModel(nn.Module):
    
    """Class for neural network
    Attributes:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
    """
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64):    
        """Initialize parameters and model architecture"""        
        super(DeepQNetModel,self).__init__()        
        self.seed = torch.manual_seed(seed)        
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
      
           
    def forward(self,state):
        """Forward propagation of neural network
        Args:
            state (vector): sized (self.state_size x batch size) with environment state data
        Returns:
            Vector sized (self.action_size x batch size) with return of a neural netowrk
        """
        x= F.relu(self.fc1(state))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        
        return x
        

    
                     
