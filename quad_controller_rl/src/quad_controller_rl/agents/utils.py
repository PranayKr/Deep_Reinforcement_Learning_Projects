import numpy as np
import random
from collections import namedtuple

# Create replay buffer
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Circular buffer for storing experience tuples"""
    
    def __init__(self, size=1000):
        """Initialize ReplayBuffer"""
        self.size = size
        self.memory = []
        self.idx = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory"""
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[idx] = e
            self.idx = (self.idx +1) % self.size
            
    def sample(self, batch_size=64):
        """Random sample of experiences"""
        return random.sample(self.memory, k=batch_size)
    
    def __len__(self):
        """Return size of internal memory"""
        return len(self.memory)


# Create Ornstein-Uhlenbeck Noise
class OUNoise:
    """Ornstein-Uhlenbeck Noise Process"""
    
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise"""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
    
    def reset(self):
        """Reset internal state to mean"""
        self.state = self.mu
        
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    

